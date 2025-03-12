import os
import argparse
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import json
import numpy as np

# Here are the COCO categories class with their IDs
COCO_CATEGORIES = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus", 7: "train", 8: "truck",
    9: "boat", 10: "traffic light", 11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow", 22: "elephant", 23: "bear",
    24: "zebra", 25: "giraffe", 27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase",
    34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite", 39: "baseball bat",
    40: "baseball glove", 41: "skateboard", 42: "surfboard", 43: "tennis racket", 44: "bottle",
    46: "wine glass", 47: "cup", 48: "fork", 49: "knife", 50: "spoon", 51: "bowl", 52: "banana",
    53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog", 59: "pizza",
    60: "donut", 61: "cake", 62: "chair", 63: "couch", 64: "potted plant", 65: "bed", 67: "dining table",
    70: "toilet", 72: "tv", 73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard", 77: "cell phone",
    78: "microwave", 79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator", 84: "book", 85: "clock",
    86: "vase", 87: "scissors", 88: "teddy bear", 89: "hair drier", 90: "toothbrush"
}


class COCODatasetDetection(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.transform = transform if transform else T.ToTensor()
        self.coco_classes = COCO_CATEGORIES
        

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_info = self.coco.loadImgs(image_id)[0]

        # Charger l'image
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir en RGB

        # Charger les annotations
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)

        # Extraire les boîtes englobantes et les labels
        boxes, labels, areas, iscrowd = [], [], [], []
        for ann in annotations:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            areas.append(w * h)
            iscrowd.append(ann.get("iscrowd", 0))  # Par défaut 0 si absent

        # Convertir en tensors PyTorch
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([image_id])

        # Appliquer les transformations
        if self.transform:
            image = self.transform(image)

        # Dictionnaire de target pour Faster R-CNN
        target = {
            "bboxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": areas,
            "iscrowd": iscrowd
        }

        return image, target

    
    def create_subset_annotations(self, num_samples=None, category_ids=None, output_file='subset_annotations.json'):
        """
        Crée un sous-ensemble d'annotations COCO et le sauvegarde dans un nouveau fichier JSON.

        Args:
            num_samples (int, optional): Nombre d'images à sélectionner.
            category_ids (list, optional): Liste d'ID de catégories pour filtrer les images.
            output_file (str): Le chemin du fichier où le sous-ensemble d'annotations sera sauvegardé.

        Returns:
            None
        """
        selected_image_ids = self.select_subset(num_samples=num_samples, category_ids=category_ids)

        # Créer un nouveau dictionnaire d'annotations COCO pour le sous-ensemble
        subset_annotations = {
            'images': [],
            'annotations': [],
            'categories': self.coco.loadCats(self.coco.getCatIds())  # Inclure toutes les catégories
        }

        # Ajouter les images sélectionnées
        for image_id in selected_image_ids:
            image_info = self.coco.loadImgs(image_id)[0]
            subset_annotations['images'].append(image_info)

            # Ajouter les annotations associées à cette image
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            annotations = self.coco.loadAnns(ann_ids)
            
            for ann in annotations:
                if category_ids and ann['category_id'] not in category_ids:
                    continue
                
                ann['image_id'] = image_id  # S'assurer que l'image_id est bien lié
                subset_annotations['annotations'].append(ann)

        # Sauvegarder le nouveau fichier d'annotations sous format JSON
        with open(output_file, 'w') as f:
            json.dump(subset_annotations, f)
            
        print(f"Le sous-ensemble d'annotations a été sauvegardé dans {output_file}")
        
    def select_subset(self, num_samples=None, category_ids=None):
        """
        Sélectionne un sous-ensemble d'images basé sur les critères donnés.
        
        Args:
            num_samples (int, optional): Nombre d'images à sélectionner aléatoirement.
            category_ids (list, optional): Liste d'ID de catégories pour filtrer les images.

        Returns:
            list: Liste des indices des images sélectionnées.
        """
        selected_image_ids = []

        for idx in range(len(self.image_ids)):
            image_id = self.image_ids[idx]
            image_info = self.coco.loadImgs(image_id)[0]

            # Charger les annotations
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            annotations = self.coco.loadAnns(ann_ids)

            # Vérifier si l'image contient l'une des catégories demandées
            if category_ids:
                labels = [ann['category_id'] for ann in annotations]
                if not any(cat in labels for cat in category_ids):
                    continue

            selected_image_ids.append(image_id)

        # Si un nombre d'échantillons est demandé, en choisir un échantillon aléatoire
        if num_samples:
            selected_image_ids = random.sample(selected_image_ids, min(num_samples, len(selected_image_ids)))

        return selected_image_ids

    def display_detection(self, image, target):
        # Generate a color by class
        random.seed(42)  # Pour avoir des couleurs stables à chaque exécution
        CLASS_COLORS = {cls_id: (random.random(), random.random(
        ), random.random()) for cls_id in COCO_CATEGORIES}

        # Convert to matplotlib format
        image_np = image.permute(1, 2, 0).numpy()

        # Display images
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(image_np)

        # Draw boxes with their corresponding labels
        for box, label in zip(target["bboxes"], target["labels"]):
            x_min, y_min, x_max, y_max = box
            class_name = COCO_CATEGORIES.get(label.item(), "Unknown")
            # Rouge par défaut si inconnu
            color = CLASS_COLORS.get(label.item(), (1, 0, 0))

            # Draw boxes
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                     linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

            # Add label with the drawn boxes
            ax.text(x_min, y_min - 5, class_name, fontsize=10, color=color,
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor=color))

        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    print("Hello Here, we passed")
    
    #dataset = COCODatasetDetection(root_dir=r"C:\Users/guill/Documents/github_projects/Datasets/coco-2017-dataset/coco2017/train2017/",
    #                               annotation_file=r"C:\Users/guill/Documents/github_projects/Datasets/coco-2017-dataset/coco2017/annotations/instances_train2017.json")
    dataset = COCODatasetDetection(root_dir=r"C:\Users/guill/Documents/github_projects/Datasets/coco-2017-dataset/coco2017/val2017/",
                                   annotation_file=r"C:\Users/guill/Documents/github_projects/Datasets/coco-2017-dataset/coco2017/annotations/instances_val2017.json")
    # create annotation dataset
    dataset.create_subset_annotations(
        num_samples=100,
        category_ids=[1, 2, 3, 4],
        output_file='subset_valid_annotations.json'
    )
    
    """image, target = dataset[5]
    dataset.display_detection(image, target)"""
    
    
