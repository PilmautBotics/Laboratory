import torch
import torch.nn as nn
import torchvision
import math

from models.fastercnn.utils import *

class ROIHead(nn.Module):
    r"""
    ROI head on top of ROI pooling layer for generating
    classification and box transformation predictions
    We have two fc layers followed by a classification fc layer
    and a bbox regression fc layer
    """
    
    def __init__(self, model_config, num_classes, in_channels):
        super(ROIHead, self).__init__()
        self.num_classes = num_classes
        self.roi_batch_size = model_config['roi_batch_size']
        self.roi_pos_count = int(model_config['roi_pos_fraction'] * self.roi_batch_size)
        self.iou_threshold = model_config['roi_iou_threshold']
        self.low_bg_iou = model_config['roi_low_bg_iou']
        self.nms_threshold = model_config['roi_nms_threshold']
        self.topK_detections = model_config['roi_topk_detections']
        self.low_score_threshold = model_config['roi_score_threshold']
        self.pool_size = model_config['roi_pool_size']
        self.fc_inner_dim = model_config['fc_inner_dim']
        
        self.fc6 = nn.Linear(in_channels * self.pool_size * self.pool_size, self.fc_inner_dim)
        self.fc7 = nn.Linear(self.fc_inner_dim, self.fc_inner_dim)
        self.cls_layer = nn.Linear(self.fc_inner_dim, self.num_classes)
        self.bbox_reg_layer = nn.Linear(self.fc_inner_dim, self.num_classes * 4)
        
        torch.nn.init.normal_(self.cls_layer.weight, std=0.01)
        torch.nn.init.constant_(self.cls_layer.bias, 0)

        torch.nn.init.normal_(self.bbox_reg_layer.weight, std=0.001)
        torch.nn.init.constant_(self.bbox_reg_layer.bias, 0)
    
    def assign_target_to_proposals(self, proposals, gt_boxes, gt_labels):
        r"""
        Given a set of proposals and ground truth boxes and their respective labels.
        Use IOU to assign these proposals to some gt box or background
        :param proposals: (number_of_proposals, 4)
        :param gt_boxes: (number_of_gt_boxes, 4)
        :param gt_labels: (number_of_gt_boxes)
        :return:
            labels: (number_of_proposals)
            matched_gt_boxes: (number_of_proposals, 4)
        """
        device = proposals.device

        # Get IOU Matrix between gt boxes and proposals
        iou_matrix = get_iou(gt_boxes, proposals)
        
        if iou_matrix.shape[0] == 0:  # If IOU matrix is empty, this is a problem
            #print("Warning: IOU matrix is empty. Check your proposals or GT boxes.")
            return torch.tensor([]).to(device), torch.tensor([]).to(device)  # Return empty tensors
        
        # For each gt box proposal find best matching gt box
        best_match_iou, best_match_gt_idx = iou_matrix.max(dim=0)
        background_proposals = (best_match_iou < self.iou_threshold) & (best_match_iou >= self.low_bg_iou)
        ignored_proposals = best_match_iou < self.low_bg_iou
        
        # Update best match of low IOU proposals to -1
        best_match_gt_idx[background_proposals] = -1
        best_match_gt_idx[ignored_proposals] = -2
        
        # Get best marching gt boxes for ALL proposals
        # Even background proposals would have a gt box assigned to it
        # Label will be used to ignore them later
        matched_gt_boxes_for_proposals = gt_boxes[best_match_gt_idx.clamp(min=0)]
        
        # Get class label for all proposals according to matching gt boxes
        labels = gt_labels[best_match_gt_idx.clamp(min=0)]
        labels = labels.to(dtype=torch.int64)
        
        # Update background proposals to be of label 0(background)
        labels[background_proposals] = 0
        
        # Set all to be ignored anchor labels as -1(will be ignored)
        labels[ignored_proposals] = -1
        
        return labels, matched_gt_boxes_for_proposals
    
    def forward(self, feat, proposals, image_shape, target):
        r"""
        Main method for ROI head that does the following:
        1. If training assign target boxes and labels to all proposals
        2. If training sample positive and negative proposals
        3. If training get bbox transformation targets for all proposals based on assignments
        4. Get ROI Pooled features for all proposals
        5. Call fc6, fc7 and classification and bbox transformation fc layers
        6. Compute classification and localization loss

        :param feat:
        :param proposals:
        :param image_shape:
        :param target:
        :return:
        """
        if self.training and target is not None:
            #print(f"Original proposals shape: {proposals.shape}")
            #print(f"Ground truth bboxes shape: {target['bboxes'][0].shape}")
        
            # Add ground truth to proposals
            proposals = torch.cat([proposals, target['bboxes'][0]], dim=0)
            #print(f"Concatenated proposals shape: {proposals.shape}")

            gt_boxes = target['bboxes'][0]
            gt_labels = target['labels'][0]
            
            labels, matched_gt_boxes_for_proposals = self.assign_target_to_proposals(proposals, gt_boxes, gt_labels)
            #print(f"Labels shape after assignment: {labels.shape}")
            #print(f"Matched ground truth boxes shape: {matched_gt_boxes_for_proposals.shape}")
            
            sampled_neg_idx_mask, sampled_pos_idx_mask = sample_positive_negative(labels,
                                                                                  positive_count=self.roi_pos_count,
                                                                                  total_count=self.roi_batch_size)
            #print(f"Sampled positive indices mask: {sampled_pos_idx_mask.shape}")
            #print(f"Sampled negative indices mask: {sampled_neg_idx_mask.shape}")
            
            sampled_idxs = torch.where(sampled_pos_idx_mask | sampled_neg_idx_mask)[0]
            #print(f"Sampled indices: {sampled_idxs.shape}")
            
            # Keep only sampled proposals
            proposals = proposals[sampled_idxs]
            labels = labels[sampled_idxs]
            matched_gt_boxes_for_proposals = matched_gt_boxes_for_proposals[sampled_idxs]
            regression_targets = boxes_to_transformation_targets(matched_gt_boxes_for_proposals, proposals)
            #print(f"Sampled proposals shape: {proposals.shape}")
            #print(f"Sampled labels shape: {labels.shape}")
            #print(f"Regression targets shape: {regression_targets.shape}")
            
            # regression_targets -> (sampled_training_proposals, 4)
            # matched_gt_boxes_for_proposals -> (sampled_training_proposals, 4)
        
        # Get desired scale to pass to roi pooling function
        # For vgg16 case this would be 1/16 (0.0625)
        size = feat.shape[-2:]
        possible_scales = []
        for s1, s2 in zip(size, image_shape):
            approx_scale = float(s1) / float(s2)
            scale = 2 ** float(torch.tensor(approx_scale).log2().round())
            possible_scales.append(scale)
        assert possible_scales[0] == possible_scales[1]
        #print(f"Possible scale for roi pooling: {possible_scales[0]}")
        
        # ROI pooling and call all layers for prediction
        proposal_roi_pool_feats = torchvision.ops.roi_pool(feat, [proposals],
                                                           output_size=self.pool_size,
                                                           spatial_scale=possible_scales[0])
        #print(f"ROI pooled features shape: {proposal_roi_pool_feats.shape}")
        
        proposal_roi_pool_feats = proposal_roi_pool_feats.flatten(start_dim=1)
        #print(f"Flattened pooled features shape: {proposal_roi_pool_feats.shape}")
        
        box_fc_6 = torch.nn.functional.relu(self.fc6(proposal_roi_pool_feats))
        box_fc_7 = torch.nn.functional.relu(self.fc7(box_fc_6))
        cls_scores = self.cls_layer(box_fc_7)
        box_transform_pred = self.bbox_reg_layer(box_fc_7)
        #print(f"cls_scores shape: {cls_scores.shape}")
        #print(f"cls_scores value: {cls_scores}")
        #print(f"box_transform_pred shape: {box_transform_pred.shape}")
        #print(f"labels detected: {labels}")
        
        # cls_scores -> (proposals, num_classes)
        # box_transform_pred -> (proposals, num_classes * 4)
        ##############################################
        
        num_boxes, num_classes = cls_scores.shape
        box_transform_pred = box_transform_pred.reshape(num_boxes, num_classes, 4)
        frcnn_output = {}
        # If no bboxes in target cls scores and localization scores is 0 
        #frcnn_output['frcnn_classification_loss'] = torch.tensor(0.0, device=cls_scores.device)
        #frcnn_output['frcnn_localization_loss'] = torch.tensor(0.0, device=box_transform_pred.device)
        if self.training and target['bboxes'].shape[1] != 0:
            classification_loss = torch.nn.functional.cross_entropy(cls_scores, labels)
            #print(f"Classification loss: {classification_loss.item()}")
                        
            # Compute localization loss only for non-background labelled proposals
            fg_proposals_idxs = torch.where(labels > 0)[0]
            #print(f"Foreground proposals indices: {fg_proposals_idxs.shape}")
            
            # Get class labels for these positive proposals
            fg_cls_labels = labels[fg_proposals_idxs]
            
            localization_loss = torch.nn.functional.smooth_l1_loss(
                box_transform_pred[fg_proposals_idxs, fg_cls_labels],
                regression_targets[fg_proposals_idxs],
                beta=1/9,
                reduction="sum",
            )
            localization_loss = localization_loss / labels.numel()
            #print(f"Localization loss: {localization_loss.item()}")
            
            frcnn_output['frcnn_classification_loss'] = classification_loss
            frcnn_output['frcnn_localization_loss'] = localization_loss
        
        if self.training:
            return frcnn_output
        else:
            device = cls_scores.device
            # Apply transformation predictions to proposals
            pred_boxes = apply_regression_pred_to_anchors_or_proposals(box_transform_pred, proposals)
            pred_scores = torch.nn.functional.softmax(cls_scores, dim=-1)
            
            # Clamp box to image boundary
            pred_boxes = clamp_boxes_to_image_boundary(pred_boxes, image_shape)
            
            # create labels for each prediction
            pred_labels = torch.arange(num_classes, device=device)
            pred_labels = pred_labels.view(1, -1).expand_as(pred_scores)
            
            # remove predictions with the background label
            pred_boxes = pred_boxes[:, 1:]
            pred_scores = pred_scores[:, 1:]
            pred_labels = pred_labels[:, 1:]
            
            #print(f"Predicted boxes shape: {pred_boxes.shape}")
            #print(f"Predicted scores shape: {pred_scores.shape}")
            #print(f"Predicted labels shape: {pred_labels.shape}")
            
            # pred_boxes -> (number_proposals, num_classes-1, 4)
            # pred_scores -> (number_proposals, num_classes-1)
            # pred_labels -> (number_proposals, num_classes-1)
            
            # batch everything, by making every class prediction be a separate instance
            pred_boxes = pred_boxes.reshape(-1, 4)
            pred_scores = pred_scores.reshape(-1)
            pred_labels = pred_labels.reshape(-1)
            #print(f"Reshaped pred_boxes shape: {pred_boxes.shape}")
            #print(f"Reshaped pred_scores shape: {pred_scores.shape}")
            #print(f"Reshaped pred_labels shape: {pred_labels.shape}")
        
            
            pred_boxes, pred_labels, pred_scores = self.filter_predictions(pred_boxes, pred_labels, pred_scores)
            frcnn_output['boxes'] = pred_boxes
            frcnn_output['scores'] = pred_scores
            frcnn_output['labels'] = pred_labels
            return frcnn_output
    
    def filter_predictions(self, pred_boxes, pred_labels, pred_scores):
        r"""
        Method to filter predictions by applying the following in order:
        1. Filter low scoring boxes
        2. Remove small size boxes∂
        3. NMS for each class separately
        4. Keep only topK detections
        :param pred_boxes:
        :param pred_labels:
        :param pred_scores:
        :return:
        """
        # remove low scoring boxes
        keep = torch.where(pred_scores > self.low_score_threshold)[0]
        pred_boxes, pred_scores, pred_labels = pred_boxes[keep], pred_scores[keep], pred_labels[keep]
        
        # Remove small boxes
        min_size = 16
        ws, hs = pred_boxes[:, 2] - pred_boxes[:, 0], pred_boxes[:, 3] - pred_boxes[:, 1]
        keep = (ws >= min_size) & (hs >= min_size)
        keep = torch.where(keep)[0]
        pred_boxes, pred_scores, pred_labels = pred_boxes[keep], pred_scores[keep], pred_labels[keep]
        
        # Class wise nms
        keep_mask = torch.zeros_like(pred_scores, dtype=torch.bool)
        for class_id in torch.unique(pred_labels):
            curr_indices = torch.where(pred_labels == class_id)[0]
            curr_keep_indices = torch.ops.torchvision.nms(pred_boxes[curr_indices],
                                                          pred_scores[curr_indices],
                                                          self.nms_threshold)
            keep_mask[curr_indices[curr_keep_indices]] = True
        keep_indices = torch.where(keep_mask)[0]
        post_nms_keep_indices = keep_indices[pred_scores[keep_indices].sort(descending=True)[1]]
        keep = post_nms_keep_indices[:self.topK_detections]
        pred_boxes, pred_scores, pred_labels = pred_boxes[keep], pred_scores[keep], pred_labels[keep]
        return pred_boxes, pred_labels, pred_scores

