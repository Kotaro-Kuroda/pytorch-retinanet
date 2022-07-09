import torch
from torchvision import transforms
import numpy as np
import config
import torchvision
import glob
import os
import tqdm
import jpeg4py
import cv2


def predict(image, model, height, width):
    img = jpeg4py.JPEG(image).decode()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(height, width),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    tensor_img = transform(img).cuda()
    tensor_img = tensor_img.unsqueeze(0)
    with torch.no_grad():
        pred = model([tensor_img])
    boxes = pred[0]['boxes']
    classes = config.dataset_class
    classes = ["__background__"] + classes
    scores = pred[0]['scores']
    indices = torchvision.ops.nms(boxes, scores, 0.3)
    boxes = boxes[indices]
    scores = scores[indices]
    for i, box in enumerate(boxes):
        score = scores[i].cpu().numpy()
        if score > 0.8:
            box = box.cpu().numpy().astype(np.int32)
            cat = int(pred[0]['labels'][i].cpu().numpy())
            label = classes[cat]
            txt = '{} {}'.format(label, str(np.round(score, 3)))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
            c = config.colors[int(cat)]
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), c, 2)
            cv2.rectangle(img, (box[0], box[1] - cat_size[1] - 2),
                          (box[0] + cat_size[0], box[1] - 2), c, -1)
            cv2.putText(img, txt, (box[0], box[1] - 2), font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    cv2.imwrite(f'/home/kotarokuroda/Documents/pytorch-retinanet/Result/{os.path.basename(image)}', img)


def main():
    image_dir = '/home/kotarokuroda/Documents/pytorch-retinanet/dataset/Test'
    list_image = glob.glob(f'{image_dir}/*.jpeg')
    model_path = '/home/kotarokuroda/Documents/pytorch-retinanet/model/model_16.pt'
    model = torch.load(model_path)
    model.cuda()
    model.eval()
    for image in tqdm.tqdm(list_image):
        predict(image, model)


if __name__ == '__main__':
    main()
