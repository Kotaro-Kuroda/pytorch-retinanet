import torch
from torchvision import transforms
import cv2
import numpy as np
import config
import torchvision


def predict(image, model_path):
    model = torch.load(model_path)
    model.cuda()
    model.eval()
    img = cv2.imread(image)
    img = cv2.resize(img, (224, 224))
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    tensor_img = transform(img).cuda()
    # img = img.unsqueeze(0)
    with torch.no_grad():
        pred = model([tensor_img])
    boxes = pred[0]['boxes']
    classes = config.dataset_class
    classes = ["__background__"] + classes
    scores = pred[0]['scores']
    print(boxes)
    indices = torchvision.ops.nms(boxes, scores, 0.5)
    boxes = boxes[indices]
    scores = scores[indices]
    for i, box in enumerate(boxes):
        score = scores[i].cpu().numpy()
        print(i)
        box = box.cpu().numpy().astype(np.int32)
        cat = int(pred[0]['labels'][i].cpu().numpy())
        label = classes[cat]
        txt = '{} {}'.format(label, str(score))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
        c = config.colors[int(cat)]
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), c, 2)
        cv2.rectangle(img, (box[0], box[1] - cat_size[1] - 2),
                      (box[0] + cat_size[0], box[1] - 2), c, -1)
        cv2.putText(img, txt, (box[0], box[1] - 2), font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.imwrite('/home/kotarokuroda/Documents/result.jpg', img)


def main():
    image = '/home/kotarokuroda/Documents/乃木坂46/Train/1f1454110705ecb16fc882d3e63ea.jpg'
    model_path = '/home/kotarokuroda/Documents/CNN/model/model_5.pt'
    predict(image, model_path)


if __name__ == '__main__':
    main()
