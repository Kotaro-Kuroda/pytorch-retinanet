import xml.etree.ElementTree as ET
from glob import glob
from torchvision import transforms
import os
import torch
import tqdm
import jpeg4py
from torchvision.transforms import functional as F
import pickle


class xml2list(object):
    def __init__(self, classes, scale):
        self.classes = classes
        self.scale = scale

    def __call__(self, xml_path):
        xml = ET.parse(xml_path).getroot()
        boxes = []
        labels = []
        size = xml.find('size')
        height = int(size.find('height').text)
        ratio = self.scale / height
        if self.scale == 1:
            ratio = 1
        num_objs = 0
        for obj in xml.iter('object'):
            label = obj.find('name').text.strip()
            if label in self.classes:
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text) * ratio
                ymin = float(bndbox.find('ymin').text) * ratio
                xmax = float(bndbox.find('xmax').text) * ratio
                ymax = float(bndbox.find('ymax').text) * ratio
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(self.classes.index(label))
                num_objs += 1

        anno = {'bboxes': boxes, 'labels': labels}

        return anno, num_objs


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, train_dir, scale, classes, multi):

        super().__init__()
        self.train_dir = train_dir
        self.scale = scale
        self.classes = classes
        self.multi = multi

        self.image_list = self._get_image_list()
        self.valid_image_list, self.list_annotaion, self.list_num_obje = self._get_valid_image_list()
        images_tensor = torch.stack(self.valid_image_list)
        self.mean = torch.mean(images_tensor, dim=(0, 2, 3))
        self.std = torch.var(images_tensor, dim=(0, 2, 3), unbiased=True)
        print(self.mean, self.std)
        with open('./args.pickle', 'wb') as f:
            pickle.dump([self.mean, self.std], f)
        self.valid_image_list = [F.normalize(image, self.mean, self.std, inplace=True) for image in self.valid_image_list]

    # 画像をリストで取得

    def _get_image_list(self):
        if self.multi:
            list_dir = [os.path.join(self.train_dir, dire) for dire in os.listdir(
                self.train_dir) if os.path.isdir(os.path.join(self.train_dir, dire))]
        else:
            list_dir = [self.train_dir]
        list_img = []
        for dire in list_dir:
            list_img += glob(dire + '/0001_0_0/*.jpg')
        return list_img

    # 取得した画像のうち、物体が写っているものだけ取得
    def _get_valid_image_list(self):
        p_bar = tqdm.tqdm(self.image_list)
        p_bar.set_description('load images')
        list_image = []
        list_annotation = []
        list_obje_num = []
        for image in p_bar:
            p_bar.set_postfix(image=image)
            xml_path = os.path.splitext(image)[0] + '.xml'
            if not os.path.exists(xml_path):
                continue
            transform_anno = xml2list(self.classes, self.scale)
            annotations, obje_num = transform_anno(xml_path)
            if obje_num > 0:
                list_image.append(self._preproc(image))
                list_annotation.append(annotations)
                list_obje_num.append(obje_num)
        return list_image, list_annotation, list_obje_num

    def _preproc(self, image):
        image = jpeg4py.JPEG(image).decode()
        h, w = image.shape[:2]
        self.width = int(self.scale / h * w)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.scale, self.width))
        ])
        return self.transform(image)

    def __getitem__(self, index):
        image = self.valid_image_list[index]
        annotations = self.list_annotaion[index]
        obje_num = self.list_num_obje[index]
        boxes = torch.as_tensor(annotations['bboxes'], dtype=torch.int64)
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
        return len(self.valid_image_list)


def dataloader(train_dir, dataset_class, batch_size, scale, multi):
    dataset = MyDataset(train_dir, scale, dataset_class, multi)
    torch.manual_seed(2020)

    def collate_fn(batch):
        return tuple(zip(*batch))

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)

    return train_dataloader
