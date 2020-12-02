from dataset import MicroscopyDataset, RandomResizedCrop, RandomRotation, RandomFlip, ToTensor, ExtendImageChannel, vis_res
from res_unet import resnet18_UNet
from loss_metric import DiceLoss

import os
import time
import json
import torch
import logging
import numpy as np
from torchvision import transforms


def do_epoch(args, epoch, net, dataset, optimizer, eval=False, K=30000):
    dataloader = torch.utils.data.DataLoader(dataset, 
                                 batch_size=args.batchsize if not eval else 1, 
                                 shuffle=not eval, 
                                 num_workers=0, 
                                 drop_last=False
                              )
    net = net.train(not eval)

    epoch_loss = 0.0
    epoch_acc = 0.0
    epoch_hem = 0.0
    for iter, data in enumerate(dataloader):
        img, mask = data
        img = img.float().cuda() if args.use_cuda else img.float().cpu()
        if mask is not None:
            mask = mask.float().cuda() if args.use_cuda else mask.float().cpu()
        pred = net(img)
        loss_dice = net.loss(pred, mask)
        loss_map = net.boundaryloss(pred, mask)
        
        if epoch+1 > (args.epoch/2):
            if not eval:
                loss_map_flat = loss_map.reshape(args.batchsize, -1)
            else:
                loss_map_flat = loss_map.reshape(1, -1)

            loss_hem = loss_map_flat.topk(K, dim=-1)[0].mean()
            w_boundary = 0.0000005
            loss = loss_map.mean() + (w_boundary * loss_hem)
            epoch_hem += loss_hem.item()
        else:
            loss = loss_dice
            
        acc = net.metric(pred.detach(), mask.detach())
        epoch_loss += loss.item()
        epoch_acc += acc.item()

        if not eval:
            loss.backward()
            optimizer.step()

        if eval:
            title = 'acc={}'
            prob = torch.sigmoid(torch.max(pred[0,:,:,:],0)[0])
            vis_res(img[0,:,:,:].permute(1, 2, 0), mask[0,0,:,:], loss_map[0,0,:,:], prob>0.5, save_path='./epoch_{}_iter_{}.png'.format(epoch, iter), title=None)

    return epoch_loss / iter, epoch_acc / iter, epoch_hem / iter
            


def run(args):
    # init model
    if args.enable_resnet_pretrain:
        net = resnet18_UNet(pretrained=not args.resume, n_class=1, input_size=256)
        if args.resume:
            net.load_state_dict(torch.load(args.resume_model_path), strict=True)
    else:
        net = resnet18_UNet(pretrained=False, n_class=1, input_size=256)
    
    net.loss = DiceLoss()
    net.boundaryloss = torch.nn.BCEWithLogitsLoss(reduction='none')
    net.metric = DiceLoss(get_coefficient=True)

    if args.use_cuda:
        net = net.cuda()
    
    # init optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # init dataloader
    train_transforms = transforms.Compose([
                            RandomRotation(),
                            RandomResizedCrop(size=args.training_patch_size),
                            RandomFlip(),
                            ToTensor(),
                            ExtendImageChannel(),
                         ])
    train_dataset = MicroscopyDataset(args.dataset_root, args.dataset_train_list, transform=train_transforms)
    val_dataset = MicroscopyDataset(args.dataset_root, args.dataset_val_list, transform=train_transforms)
    logging.info('Train set size:{}, Val set size:{}'.format(len(train_dataset), len(val_dataset)))

    logging.info('Start training...')
    for epoch in range(args.epoch):
        time1 = time.time()
        loss, acc, hem = do_epoch(args, epoch, net, train_dataset, optimizer)
        time2 = time.time()
        logging.info('epoch [{}/{}], time elapse={:.2f}, loss={:.4f}, train_acc={:.4f}, hem_loss={:.4f}'.format(epoch, args.epoch, time2-time1, loss, acc, hem))

        if epoch % args.eval_freq == 0:
            logging.info('Evaluate model...')
            loss, ac, hem = do_epoch(args, epoch, net, val_dataset, None, eval=True)
            logging.info('val_loss={:.4f}, val_acc={:.4f}, hem_loss={:.4f}'.format(loss, acc, hem))
        
    logging.info('Finish training...')


if __name__ == '__main__':
    def bool_str(x):
        return str(x).lower() in ['True', 'true', '1']   
    
    import configargparse
    parser = configargparse.ArgParser()
    
    parser.add_argument('--dataset-root', type=str, required=True)
    parser.add_argument('--training-patch-size', default=512, type=int, help='image crop size during training')
    parser.add_argument('--dataset-train-list', type=str, required=True)
    parser.add_argument('--dataset-val-list', type=str, required=True)
    
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epoch', default=10, type=int, help='learning epochs')
    parser.add_argument('--batchsize', default=5, type=int, help='batch size during training')
    parser.add_argument('--eval-freq', default=1, type=int, help='to evaluate and visualize at every X epochs')
    
    parser.add_argument('--enable-resnet-pretrain', type=bool_str, required=True)
    parser.add_argument('--resume', default=False, type=bool_str, help='load weights from a specified model')
    parser.add_argument('--resume-model-path', type=str)

    parser.add_argument('--use-cuda', default=True, type=bool_str)
    
    args = parser.parse_args()
    args.training_patch_size = [args.training_patch_size, args.training_patch_size]

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(filename)s:%(lineno)s %(levelname)s %(message)s')
    logging.info(json.dumps(vars(args), indent=4))

    run(args)
