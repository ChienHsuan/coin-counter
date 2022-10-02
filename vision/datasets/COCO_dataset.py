import os
import torch
import torch.utils.data
import torchvision
import cv2
import numpy as np

from pycocotools.coco import COCO


class COCODataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.coco = COCO(annotation)
        self.class_names = ['background', 'person']
        self.ids = list(sorted(self.coco.getImgIds(catIds=1))) #class: person

    def __getitem__(self, index):
        # Image ID
        img_id = self.ids[index]

        # List: get annotation id
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=1)

        # Dictionary: target coco_annotation file for an image
        coco_annotation = self.coco.loadAnns(ann_ids)

        # path for input image
        path = self.coco.loadImgs(img_id)[0]['file_name']

        # open the input image
        img = cv2.imread(str(os.path.join(self.root, path)))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = np.array(boxes, dtype=np.float32)

        # Labels
        labels = np.ones((num_objs,), dtype=np.int64)

        if self.transform is not None:
            img, boxes, labels = self.transform(img, boxes, labels)
        if self.target_transform is not None:
            boxes, labels = self.target_transform(boxes, labels)

        return img, boxes, labels

    def __len__(self):
        return len(self.ids)

