import numpy as np
import torch
import os
import retinanet_model
import config
<<<<<<< HEAD
import tqdm
import mydataset
=======
import mydataset
import matplotlib.pyplot as plt
>>>>>>> f6aaae0d6abbb870008ee5aa006a42c68486c29c
from torch.utils.tensorboard import SummaryWriter
import argparse
import warnings
from distutils.util import strtobool
from sklearn.model_selection import ShuffleSplit
from torch.utils.data.dataset import Subset
import torchvision
import tqdm
import datetime
from torch.optim import lr_scheduler
warnings.simplefilter('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a Faster R-CNN')
    parser.add_argument('--train_dir', default='/home/localuser/Documents/technical_sample/Train_CUT_NG', help='training dataset')
    parser.add_argument('--epoch', '-e', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--batch', '-b', type=int, default=16, help='batch size')
    parser.add_argument('--scale', '-s', type=int, default=1365, help='scaling size of the image')
    parser.add_argument('--nn_model', '-n', choices=('resnet50', 'mobilenet_v3_large', 'retinanet'), default='retinanet', type=str, help='neural network model to use: resnet50, mobilenet_v3_large')
    parser.add_argument('--multi', '-m', type=strtobool, default="True", help='whether direcotries of the datasets are multiple or not')
    args = parser.parse_args()
    return args


def collate_fn(batch):
    return tuple(zip(*batch))


def get_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Calculate area under precision/recall curve.
    Args:
      recalls:
      precisions:
    Returns:
    """
    # correct AP calculation
    # first append sentinel values at the end
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    # to calculate area under PR curve, look for points where X axis (recall) changes value
    i = np.where(recalls[1:] != recalls[:-1])[0]
    # and sum (\Delta recall) * prec
    ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
    return ap


def train(model, train_loader, optimizer):
    loss_epo = []
    pbar = tqdm.tqdm(train_loader)
    scaler = torch.cuda.amp.GradScaler()
    model.train()
    for i, batch in enumerate(pbar):
        images, targets = batch
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()}
                   for t in targets]
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        loss_epo.append(loss_value)
        pbar.set_postfix(loss=str(loss_value))
        scaler.update()
    return loss_epo


def validation(model, val_loader):
    model.eval()
    precision_list = []
    recall_list = []
    correct = 0
    num_obj = 0
    k = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(val_loader):
            images, targets = batch
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()}
                       for t in targets]
            prediction = model(images)
            scores = prediction[0]['scores']
            pred_box = prediction[0]['boxes']
            target_box = targets[0]['boxes']
            pred_label = prediction[0]['labels']
            target_label = targets[0]['labels']
            iou_matrix = torchvision.ops.box_iou(target_box, pred_box)
            num_obj += len(target_label)
            for j in range(len(pred_box)):
                k += 1
                index = torch.argmax(iou_matrix[:, j])
                if iou_matrix[index, j] >= 0.5 and pred_label[j] == target_label[index]:
                    correct += 1
                precision_list.append(correct / (k))
                recall_list.append(correct)
    precision_list = [max(precision_list[i:])
                      for i in range(len(recall_list))]
    recall_list = np.array(recall_list) / num_obj
    precision_list = np.array(precision_list)
    return recall_list, precision_list


def cross_validation(args, model, dataset, optimizer, model_save_dir):
    kf = ShuffleSplit(n_splits=10, test_size=0.10, random_state=0)
    writer = SummaryWriter()
<<<<<<< HEAD
    dataset_class = config.dataset_class
    model = retinanet.model(dataset_class)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9, weight_decay=0.00005)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=1e-3)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    num = len(os.listdir(save_dir))
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.train()
    train_loader = mydataset.dataloader(
        train_dir, dataset_class, batch_size, height, width)
    for e in tqdm.tqdm(range(epoch)):
        scheduler.step()
        loss_epo = train(train_loader, model, optimizer, device)
        loss = np.mean(loss_epo)
        writer.add_scalar('training loss', loss, e)
        torch.save(model, f'{save_dir}/model_{str(num + 1)}.pt')
    torch.save(model, f'{save_dir}/model_{str(num + 1)}.pt')
=======
    base_dir = os.getcwd()
    loss_list = []
    best_loss = float('inf')
    for _fold, (train_index, val_index) in enumerate(kf.split(dataset)):
        train_dataset = Subset(dataset, train_index)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)
        val_dataset = Subset(dataset, val_index)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)
        for epoch in tqdm.tqdm(range(args.epoch)):
            print(_fold * args.epoch + epoch + 1)
            loss_epo = train(model, train_loader, optimizer)
            fig = plt.figure(figsize=(10, 10))
            recall_list, precision_list = validation(model, val_loader)
            ap = get_ap(recall_list, precision_list)
            plt.plot(recall_list, precision_list)
            loss_list.append(np.mean(loss_epo))
            num_iter = _fold * args.epoch + epoch + 1
            writer.add_scalar('Train Loss', np.mean(loss_epo), num_iter)
            writer.add_figure('precision vs recall', fig, num_iter)
            writer.add_scalar('average precision', ap, num_iter)
            if num_iter % 10 == 0:
                torch.save(model.state_dict(), f'{model_save_dir}/model{str(num_iter)}.pt')
        if np.mean(loss_epo) < best_loss:
            best_loss = np.mean(loss_epo)
            torch.save(model.state_dict(), f'{model_save_dir}/best.pt')
    plt.plot(loss_list, 'ro-')
    plt.savefig(base_dir + '/loss.jpg')
    torch.save(model.state_dict(), f'{model_save_dir}/model.pt')


def train_model(args, model, classes, optimizer, model_path):
    writer = SummaryWriter()
    base_dir = os.getcwd()
    torch.backends.cudnn.benchmark = True
    loss_list = []
    print('loading')
    train_loader = mydataset.dataloader(args.train_dir, classes, args.batch, args.scale, args.multi)
    print('complete')
    pbar = tqdm.tqdm(range(args.epoch))
    for epoch in pbar:
        pbar.set_postfix(epoch=str(epoch + 1))
        loss_epo = train(model, train_loader, optimizer)
        loss_list.append(np.mean(loss_epo))
        writer.add_scalar('Train Loss', np.mean(loss_epo), epoch + 1)
        torch.save(model.state_dict(), model_path)
    plt.plot(loss_list, 'ro-')
    plt.savefig(base_dir + '/loss.jpg')
>>>>>>> f6aaae0d6abbb870008ee5aa006a42c68486c29c


def main(args):
    base_dir = os.getcwd()
    dataset_class = config.dataset_class
    model = retinanet_model.poolformer_backbone_model(len(dataset_class) + 1)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=1e-3, momentum=0.9, weight_decay=1 * 1e-4)
    model_save_dir = os.path.join(base_dir, 'models')
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)
    model_save_dir = os.path.join(model_save_dir, args.nn_model)
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')
    model_save_dir = os.path.join(model_save_dir, now)
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)
    # train_model(args, model, dataset_class, optimizer, device, model_path)
    dataset = mydataset.MyDataset(args.train_dir, args.scale, dataset_class, args.multi)
    cross_validation(args, model, dataset, optimizer, model_save_dir)


if __name__ == '__main__':
    args = parse_args()
    main(args)
