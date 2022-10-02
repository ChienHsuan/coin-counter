import os
import xml.etree.ElementTree as ET

import torch
import torch.utils.data
from PIL import Image


class CoinDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_sets_file = os.path.join(self.root, 'images.txt')
        labels_file = os.path.join(self.root, 'labels.txt')
        if os.path.isfile(labels_file):
            class_string = ""
            with open(labels_file, 'r') as infile:
                for line in infile:
                    class_string += line.rstrip()

            classes = class_string.split(',')
            classes  = [ elem.replace(" ", "") for elem in classes]
            self.class_names = tuple(classes)
            self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}
        else:
            raise NameError('ERROR: labels file error')
        
        self.all_samples = []
        self._read_image_ids(self.image_sets_file)
        
    def __getitem__(self, index):
        sample = self.all_samples[index]
        img = self._read_image(sample)
        label = sample[1]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def _read_image_ids(self, image_sets_file):
        with open(image_sets_file) as f:
            for line in f:
                image_id = line.rstrip()
                annotation_file = os.path.join(self.root, f'{image_id}.xml')
                objects = ET.parse(annotation_file).findall("object")
                for object in objects:
                    class_name = object.find('name').text.lower().strip()

                    if class_name in self.class_dict:
                        bbox = object.find('bndbox')
                        x1 = float(bbox.find('xmin').text) - 1
                        y1 = float(bbox.find('ymin').text) - 1
                        x2 = float(bbox.find('xmax').text) - 1
                        y2 = float(bbox.find('ymax').text) - 1
                        label = self.class_dict[class_name]
                        
                        self.all_samples.append([image_id, label, x1, y1, x2, y2])
    
    def _read_image(self, sample):
        img_path = os.path.join(self.root, f'{sample[0]}.jpg')
        img = Image.open(str(img_path)).copy()
        img = img.crop((sample[2], sample[3], sample[4], sample[5]))
        img = img.convert('RGB')
        return img

    def __len__(self):
        return len(self.all_samples)
