import torch
import numpy as np
import argparse
import retinanet
import config
import tqdm
import dataloader
from torch.utils.tensorboard import SummaryWriter
import os
from torch.optim import lr_scheduler


def arg_parser():
    parser = argparse.ArgumentParser(description='Training RetinaNet')
    parser.add_argument('--dataset', type=str, default='/home/kotarokuroda/Documents/乃木坂46/Train/', help='path to the dataset')
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--height', type=int, default=224, help='resized height of image')
    parser.add_argument('--width', type=int, default=224, help='resized width of image')
    parser.add_argument('--epoch', type=int, default=100, help='number of iteration to train')
    parser.add_argument('--save_dir', type=str, default='/home/kotarokuroda/Documents/pytorch-retinanet/model/', help='save directory of the trained model')
    args = parser.parse_args()
    return args


def train(train_loader, model, optimizer, device):
    loss_epo = []
    pbar = tqdm.tqdm(train_loader)
    for i, batch in enumerate(pbar):
        images, targets = batch
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()}
                   for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        loss_epo.append(loss_value)
        pbar.set_postfix(loss=loss_value)
    return loss_epo


def process(train_dir, height, width, batch_size, epoch, save_dir):
    writer = SummaryWriter()
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
    train_loader = dataloader.dataloader(
        train_dir, dataset_class, batch_size, height, width)
    for e in tqdm.tqdm(range(epoch)):
        scheduler.step()
        loss_epo = train(train_loader, model, optimizer, device)
        loss = np.mean(loss_epo)
        writer.add_scalar('training loss', loss, e)
        torch.save(model, f'{save_dir}/model_{str(num + 1)}.pt')
    torch.save(model, f'{save_dir}/model_{str(num + 1)}.pt')


def main():
    args = arg_parser()
    process(args.dataset, args.height, args.width, args.batch_size, args.epoch, args.save_dir)


if __name__ == '__main__':
    main()
