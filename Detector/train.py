import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random
import json,os
from tqdm import tqdm
import mlflow
import mlflow.pytorch

from datasets.cocodataset import COCODatasetDetection
from models.fastercnn.faster_rcnn import FasterRCNN
from evaluate_model import evaluate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(
        description='Arguments for training configuration')
    parser.add_argument('-c', '--config', default='config/voc.json', type=str)
    args = parser.parse_args()

    return args

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def main(args):
    
    # Set tracking mlflow folder 
    mlflow.set_experiment('Pytorch-FasterRCNN-test')

    # Read the config file #
    with open(args.config, 'r') as f:
        config = json.load(f)
        
    seed_everything(config['training']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset_cfg = config['dataset']['train']
    valid_dataset_cfg = config['dataset']['valid']
    model_config = config['model']
    train_config = config['training']

    #_____________ Create Dataloader from coco dataset wrapper
    coco_train_dataset = COCODatasetDetection(
        root_dir=train_dataset_cfg["root_dir"], annotation_file=train_dataset_cfg["annotation_file"])
    coco_valid_dataset = COCODatasetDetection(
        root_dir=valid_dataset_cfg["root_dir"], annotation_file=valid_dataset_cfg["annotation_file"])
    
    train_dataset = DataLoader(coco_train_dataset,
                               batch_size=1,
                               shuffle=True,
                               num_workers=4)
    valid_dataset = DataLoader(coco_valid_dataset,
                               batch_size=1,
                               shuffle=False,
                               num_workers=4)
    
    #_____________ Create instance model
    model = FasterRCNN(model_config,
                                   num_classes=train_dataset_cfg['num_classes'])
    
    
    #_____________ Define training parameters from config file
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
    optimizer = torch.optim.SGD(lr=train_config['learning_rate'],
                                params=filter(lambda p: p.requires_grad,
                                              model.parameters()),
                                weight_decay=train_config['weight_decay'],
                                momentum=train_config['momentum'])
    scheduler = MultiStepLR(
        optimizer, milestones=train_config['lr_scheduler_steps'], gamma=0.1)

    acc_steps = train_config['acc_steps']
    num_epochs = train_config['num_epochs']
    step_count = 1
    
    save_dir = config['training']['task_name']
    os.makedirs(save_dir, exist_ok=True)
    
    #_____________ LOOP OVER EPOCHS FOR TRAINING
    with mlflow.start_run():
        mlflow.log_params(config['training'])
        for epoch in range(num_epochs):
            model.train()
            model.to(device)
    
            rpn_classification_losses = []
            rpn_localization_losses = []
            frcnn_classification_losses = []
            frcnn_localization_losses = []
            optimizer.zero_grad()

            for im, target in tqdm(train_dataset):
                im = im.float().to(device)
                target['bboxes'] = target['bboxes'].float().to(device)
                target['labels'] = target['labels'].long().to(device)
                
                # Make the forward pass
                rpn_output, frcnn_output = model(im, target)

                # Compute individual losses
                rpn_loss = rpn_output['rpn_classification_loss'] + \
                    rpn_output['rpn_localization_loss']
                frcnn_loss = frcnn_output['frcnn_classification_loss'] + \
                    frcnn_output['frcnn_localization_loss']
                    
                # Combine losses
                loss = rpn_loss + frcnn_loss

                # Accumulate losses for logging
                rpn_classification_losses.append(
                    rpn_output['rpn_classification_loss'].item())
                rpn_localization_losses.append(
                    rpn_output['rpn_localization_loss'].item())
                frcnn_classification_losses.append(
                    frcnn_output['frcnn_classification_loss'].item())
                frcnn_localization_losses.append(
                    frcnn_output['frcnn_localization_loss'].item())
                
                # Average loss over accumulation steps
                loss = loss / acc_steps
                
                # Backpropagate loss
                loss.backward()
                
                # Update weights after accumulating gradients for accumulation steps iterations
                if step_count % acc_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    
                step_count += 1
                
            print('Finished epoch {}'.format(epoch))
            optimizer.step()
            optimizer.zero_grad()
            torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                                    train_config['ckpt_name']))
            
            # Display loss statistics
            loss_output = ''
            loss_output += 'RPN Classification Loss : {:.4f}'.format(
                np.mean(rpn_classification_losses))
            loss_output += ' | RPN Localization Loss : {:.4f}'.format(
                np.mean(rpn_localization_losses))
            loss_output += ' | FRCNN Classification Loss : {:.4f}'.format(
                np.mean(frcnn_classification_losses))
            loss_output += ' | FRCNN Localization Loss : {:.4f}'.format(
                np.mean(frcnn_localization_losses))
            print(loss_output)
            
            # update the learning rate scheduler
            scheduler.step()
            
            # log metrics with ml flow
            mlflow.log_metrics({"rpn_classification_loss": np.mean(rpn_classification_losses),
                "rpn_localization_loss": np.mean(rpn_localization_losses),
                "frcnn_classification_loss": np.mean(frcnn_classification_losses),
                "frcnn_localization_loss": np.mean(frcnn_localization_losses),
                "lr": optimizer.param_groups[0]['lr']
            }, step=epoch)
            
            # 🚀 Évaluation toutes les `eval_interval` époques
            if epoch % valid_dataset_cfg["epoch_interval"] == 0 or epoch == num_epochs - 1:
                print(f"Starting evaluation at epoch {epoch}...")
                
                mean_ap, class_aps = evaluate(model, valid_dataset, coco_valid_dataset.coco_classes, device)
                for cls, ap in class_aps.items():
                    print(f'AP for {cls}: {ap:.4f}')
                    
                # Logging MLflow
                mlflow.log_metric("mAP", mean_ap, step=epoch)
                for cls, ap in class_aps.items():
                    mlflow.log_metric(f"AP_class_{cls}", ap.item(), step=epoch)

        print('Done Training...')
    

if __name__ == '__main__':

    # Get arguments from config path
    args = parse_args()
    main(args)
