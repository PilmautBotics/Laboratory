import torch
import numpy as np
import cv2
import os
import random
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader


class ObjectDetectionEvaluator:
    def __init__(self, dataloader, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataloader = self.load_config(config_path)
        
    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def load_model_and_dataset(self):
        dataset_config = self.config['dataset_params']
        model_config = self.config['model_params']
        train_config = self.config['train_params']
        
        # Fix seed for reproducibility
        seed = train_config['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if self.device == 'cuda':
            torch.cuda.manual_seed_all(seed)
        
        dataset = self.dataset_class('test', im_dir=dataset_config['im_test_path'], ann_dir=dataset_config['ann_test_path'])
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        model = self.model_class(model_config, num_classes=dataset_config['num_classes'])
        model.to(self.device)
        model.eval()
        model.load_state_dict(torch.load(os.path.join(train_config['task_name'], train_config['ckpt_name']),
                                         map_location=self.device))
        return model, dataset, dataloader
    
    def infer_sample(self, num_samples=10, save_path='samples'):
        os.makedirs(save_path, exist_ok=True)
        self.model.roi_head.low_score_threshold = 0.7
        
        for i in tqdm(range(num_samples)):
            idx = random.randint(0, len(self.dataset))
            image, target, fname = self.dataset[idx]
            image = image.unsqueeze(0).float().to(self.device)
            
            output = self.model(image, None)
            boxes, labels, scores = output['boxes'], output['labels'], output['scores']
            
            self.visualize_and_save(image.cpu().numpy(), boxes, labels, scores, fname, save_path, i)
    
    def visualize_and_save(self, image, boxes, labels, scores, fname, save_path, index):
        image = cv2.imread(fname)
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.detach().cpu().numpy())
            label = self.dataset.idx2label[labels[idx].item()]
            score = scores[idx].item()
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(image, f'{label}: {score:.2f}', (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(save_path, f'output_{index}.jpg'), image)
    
    def evaluate(self, iou_threshold=0.5):
        gt_boxes, pred_boxes = [], []
        for image, target, fname in tqdm(self.dataloader):
            image = image.float().to(self.device)
            output = self.model(image, None)
            
            # Format predictions
            boxes, labels, scores = output['boxes'], output['labels'], output['scores']
            pred_dict = {self.dataset.idx2label[label.item()]: [] for label in labels}
            for i in range(len(boxes)):
                pred_dict[self.dataset.idx2label[labels[i].item()]].append(boxes[i].tolist() + [scores[i].item()])
            
            # Format ground truth
            gt_dict = {self.dataset.idx2label[label.item()]: [] for label in target['labels'][0]}
            for i in range(len(target['bboxes'][0])):
                gt_dict[self.dataset.idx2label[target['labels'][0][i].item()]].append(target['bboxes'][0][i].tolist())
            
            pred_boxes.append(pred_dict)
            gt_boxes.append(gt_dict)
        
        mean_ap, class_aps = self.compute_map(pred_boxes, gt_boxes, iou_threshold)
        print(f'mAP: {mean_ap:.4f}')
        for cls, ap in class_aps.items():
            print(f'AP for {cls}: {ap:.4f}')
    
    def compute_map(self, det_boxes, gt_boxes, iou_threshold=0.5, method='area'):
        # det_boxes = [
        #   {
        #       'person' : [[x1, y1, x2, y2, score], ...],
        #       'car' : [[x1, y1, x2, y2, score], ...]
        #   }
        #   {det_boxes_img_2},
        #   ...
        #   {det_boxes_img_N},
        # ]
        #
        # gt_boxes = [
        #   {
        #       'person' : [[x1, y1, x2, y2], ...],
        #       'car' : [[x1, y1, x2, y2], ...]
        #   },
        #   {gt_boxes_img_2},
        #   ...
        #   {gt_boxes_img_N},
        # ]
        
        gt_labels = {cls_key for im_gt in gt_boxes for cls_key in im_gt.keys()}
        gt_labels = sorted(gt_labels)
        all_aps = {}
        # average precisions for ALL classes
        aps = []
        for idx, label in enumerate(gt_labels):
            # Get detection predictions of this class
            cls_dets = [
                [im_idx, im_dets_label] for im_idx, im_dets in enumerate(det_boxes)
                if label in im_dets for im_dets_label in im_dets[label]
            ]
            
            # cls_dets = [
            #   (0, [x1_0, y1_0, x2_0, y2_0, score_0]),
            #   ...
            #   (0, [x1_M, y1_M, x2_M, y2_M, score_M]),
            #   (1, [x1_0, y1_0, x2_0, y2_0, score_0]),
            #   ...
            #   (1, [x1_N, y1_N, x2_N, y2_N, score_N]),
            #   ...
            # ]
            
            # Sort them by confidence score
            cls_dets = sorted(cls_dets, key=lambda k: -k[1][-1])
            
            # For tracking which gt boxes of this class have already been matched
            gt_matched = [[False for _ in im_gts[label]] for im_gts in gt_boxes]
            # Number of gt boxes for this class for recall calculation
            num_gts = sum([len(im_gts[label]) for im_gts in gt_boxes])
            tp = [0] * len(cls_dets)
            fp = [0] * len(cls_dets)
            
            # For each prediction
            for det_idx, (im_idx, det_pred) in enumerate(cls_dets):
                # Get gt boxes for this image and this label
                im_gts = gt_boxes[im_idx][label]
                max_iou_found = -1
                max_iou_gt_idx = -1
                
                # Get best matching gt box
                for gt_box_idx, gt_box in enumerate(im_gts):
                    gt_box_iou = get_iou(det_pred[:-1], gt_box)
                    if gt_box_iou > max_iou_found:
                        max_iou_found = gt_box_iou
                        max_iou_gt_idx = gt_box_idx
                # TP only if iou >= threshold and this gt has not yet been matched
                if max_iou_found < iou_threshold or gt_matched[im_idx][max_iou_gt_idx]:
                    fp[det_idx] = 1
                else:
                    tp[det_idx] = 1
                    # If tp then we set this gt box as matched
                    gt_matched[im_idx][max_iou_gt_idx] = True
            # Cumulative tp and fp
            tp = np.cumsum(tp)
            fp = np.cumsum(fp)
            
            eps = np.finfo(np.float32).eps
            recalls = tp / np.maximum(num_gts, eps)
            precisions = tp / np.maximum((tp + fp), eps)

            if method == 'area':
                recalls = np.concatenate(([0.0], recalls, [1.0]))
                precisions = np.concatenate(([0.0], precisions, [0.0]))
                
                # Replace precision values with recall r with maximum precision value
                # of any recall value >= r
                # This computes the precision envelope
                for i in range(precisions.size - 1, 0, -1):
                    precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
                # For computing area, get points where recall changes value
                i = np.where(recalls[1:] != recalls[:-1])[0]
                # Add the rectangular areas to get ap
                ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
            elif method == 'interp':
                ap = 0.0
                for interp_pt in np.arange(0, 1 + 1E-3, 0.1):
                    # Get precision values for recall values >= interp_pt
                    prec_interp_pt = precisions[recalls >= interp_pt]
                    
                    # Get max of those precision values
                    prec_interp_pt = prec_interp_pt.max() if prec_interp_pt.size > 0.0 else 0.0
                    ap += prec_interp_pt
                ap = ap / 11.0
            else:
                raise ValueError('Method can only be area or interp')
            if num_gts > 0:
                aps.append(ap)
                all_aps[label] = ap
            else:
                all_aps[label] = np.nan
        # compute mAP at provided iou threshold
        mean_ap = sum(aps) / len(aps)
        return mean_ap, all_aps


# Utilisation :
if __name__ == '__main__':
    from model.faster_rcnn import FasterRCNN  # Exemple de modèle
    from dataset.voc import VOCDataset  # Exemple de dataset
    
    evaluator = ObjectDetectionEvaluator('config/voc.yaml', FasterRCNN, VOCDataset)
    evaluator.infer_sample()
    evaluator.evaluate()
