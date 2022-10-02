import sys
import time

import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models

from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor


if len(sys.argv) < 5:
    print('Usage: python run_ssd_example.py <net type>  <model path> <label path> <image path>')
    sys.exit(0)
net_type = sys.argv[1]
model_path = sys.argv[2]
label_path = sys.argv[3]
image_path = sys.argv[4]

coins_value = {0:1, 1:5, 2:10, 3:50}

class_names = [name.strip() for name in open(label_path).readlines()]

if net_type == 'mb2-ssd-lite':
    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True, device=torch.device('cpu'))
else:
    print(f"The net type is not supported : {net_type}.")
    sys.exit(1)

net.load(model_path)

if net_type == 'mb2-ssd-lite':
    predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200, device=torch.device('cpu'))

device = torch.device('cpu')
test_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

coins_value = {0:1, 1:5, 2:10, 3:50}

# mobilenet v2
mobilenetv2_model = models.mobilenet_v2(pretrained=False, num_classes=len(coins_value.keys()))
mobilenetv2_pretrained_weight = torch.load('trained-models/mobilenetv2.pth', map_location=device)
mobilenetv2_model.load_state_dict(mobilenetv2_pretrained_weight)
mobilenetv2_model.eval()

with torch.no_grad():
    start = time.time()

    orig_image = cv2.imread(image_path)
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    boxes, labels, probs = predictor.predict(image, 10, 0.2)

    value_labels = []
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        crop = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        crop = Image.fromarray(crop)
        img = test_transform(crop)
        pred = mobilenetv2_model(img.unsqueeze(0))
        value_labels.append(coins_value[torch.argmax(pred).item()])
    end = time.time()

    for i in range(boxes.size(0)):
        box = boxes[i, :]
        cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 4)
        label = f'{value_labels[i]}'
        cv2.putText(orig_image, label,
                    (int(box[0]) + 20, int(box[1]) + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 255, 255),
                    2)  # line type
    path = "mobilenetv2_output.jpg"
    cv2.imwrite(path, orig_image)
    print(f'Cost(mobilenetv2): {end-start} s')


# resnet 50
resnet50_model = models.resnet50(pretrained=False, num_classes=len(coins_value.keys()))
resnet50_pretrained_weight = torch.load('trained-models/resnet50.pth', map_location=device)
resnet50_model.load_state_dict(resnet50_pretrained_weight)
resnet50_model.eval()

with torch.no_grad():
    start = time.time()
    orig_image = cv2.imread(image_path)
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    boxes, labels, probs = predictor.predict(image, 10, 0.2)

    value_labels = []
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        crop = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        crop = Image.fromarray(crop)
        img = test_transform(crop)
        pred = resnet50_model(img.unsqueeze(0))
        value_labels.append(coins_value[torch.argmax(pred).item()])
    end = time.time()

    for i in range(boxes.size(0)):
        box = boxes[i, :]
        cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 4)
        label = f'{value_labels[i]}'
        cv2.putText(orig_image, label,
                    (int(box[0]) + 20, int(box[1]) + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 255, 255),
                    2)  # line type
    path = "resnet50_output.jpg"
    cv2.imwrite(path, orig_image)
    print(f'Cost(resnet50): {end-start} s')
