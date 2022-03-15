import xml.etree.ElementTree as ET
import torch
import glob
from torchvision import transforms
import cv2
import numpy as np
import os


class XML2List:

    def __init__(self, classes):
        self.classes = classes

    def __call__(self, xml_path):
        xml = ET.parse(xml_path).getroot()
        boxes = np.empty((0, 4))
        labels = []
        for zz, obj in enumerate(xml.iter('object')):
            label = obj.find('name').text
            if label in self.classes:
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                box = np.array([xmin, ymin, xmax, ymax])
                boxes = np.vstack((boxes, box))
                labels.append(self.classes.index(label))
            else:
                continue
        num_objs = zz + 1
        anno = {'bboxes': boxes, 'labels': labels}
        return anno, num_objs


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, image_dir, height, width, classes):
        super().__init__()
        self.image_dir = image_dir
        self.height = height
        self.width = width
        self.classes = classes
        self.list_image = glob.glob(f'{self.image_dir}/*.jpg')
        self.list_image_array = [(self._read_image(image)) for image in self.list_image]

    def _read_image(self, image):
        img = cv2.imread(image)
        h, w = img.shape[:2]
        img = cv2.resize(img, dsize=(self.width, self.height))
        return img, h, w

    def __getitem__(self, index):

        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        image_name = self.list_image[index]
        path_xml = f'{os.path.splitext(image_name)[0]}.xml'
        image, h, w = self.list_image_array[index]

        transform_anno = XML2List(self.classes)
        annotations, obje_num = transform_anno(path_xml)
        boxes = annotations['bboxes']
        boxes = boxes * np.array([self.width / w, self.height / h, self.width / w, self.height / h])
        boxes = torch.as_tensor(boxes, dtype=torch.int64)
        image = transform(image)
        labels = torch.as_tensor(annotations['labels'], dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)
        iscrowd = torch.zeros((obje_num,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels + 1
        target["image_id"] = torch.tensor([index])
        target["area"] = area
        target["iscrowd"] = iscrowd
        return image, target

    def __len__(self):
        return len(self.list_image)


def dataloader(train_dir, dataset_class, batch_size, height, width):
    dataset = MyDataset(train_dir, height, width, dataset_class)
    torch.manual_seed(2020)

    def collate_fn(batch):
        return tuple(zip(*batch))

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return train_dataloader
