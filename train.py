import shutil
import sys

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import os
import yaml
import time
# from tensorboard_logger import configure, log_value
from test_model import start_test, test

from train_config import parse_args
from function import data_config, optimizer_function, load_checkpoint, lr_scheduler, AverageMeter, save_checkpoint, \
    gradual_warmup, fix_seed, Logger
from models.model import Network
from CMPM import Loss
import matplotlib.pyplot as plt

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(epoch, train_loader, network, opitimizer, compute_loss, args, checkpoint_dir):
    train_loss = AverageMeter()
    # switch to train mode
    network.train()

    for step, (images, captions, labels, mask) in enumerate(train_loader):
        images = images.to(device)
        captions = captions.to(device)
        labels = labels.to(device)
        mask = mask.to(device)
        opitimizer.zero_grad()
        interval = images.shape[0]
        # compute loss
        img_f5, \
        txt_f5= network(interval,images, captions, mask)
        loss = compute_loss( img_f5,\
                   txt_f5,  labels)
        train_loss.update(loss, images.shape[0])

        # graduate
        loss.backward()
        opitimizer.step()

        if step % 10== 0:
            print(
                "Train Epoch:[{}/{}] iteration:[{}/{}] cmpm_loss:{:.4f}".format(epoch + 1, args.num_epoches, step,
                                                                                len(train_loader), train_loss.avg,))
    state = {"epoch": epoch + 1,
             "state_dict": network.state_dict(),
             "W": compute_loss.W
             }

    save_checkpoint(state, epoch+1, checkpoint_dir)
    return train_loss.avg
def main(network, dataloader, compute_loss, optimizer, scheduler, start_epoch, args, checkpoint_dir):
    ac_t2i_top1_best = 0.0
    best_epoch = 0
    start = time.time()
    epochs = []
    acc = []
    loss = []
    for epoch in range(start_epoch, args.num_epoches):
        print("**********************************************************")

        if epoch < args.warm_epoch:
            print('learning rate warm_up')
            if args.optimizer == 'sgd':
                optimizer = gradual_warmup(epoch, args.sgd_lr, optimizer, epochs=args.warm_epoch)
            else:
                optimizer = gradual_warmup(epoch, args.adam_lr, optimizer, epochs=args.warm_epoch)

        train_loss=train(epoch, dataloader['train'], network, optimizer, compute_loss, args, checkpoint_dir)

        scheduler.step()

        Epoch_time = time.time() - start
        start = time.time()
        print('Epoch_training complete in {:.0f}m {:.0f}s'.format(
            Epoch_time // 60, Epoch_time % 60))
        for param in optimizer.param_groups:
            print('lr:{}'.format(param['lr']))

        ac_top1_t2i, ac_top5_t2i, ac_top10_t2i, mAP = test(dataloader['val'], network, args)

        print('train_loss:{}'.format(train_loss.cuda().data.cpu().numpy()))
        print('Epoch:{}'.format(epoch+1))
        print('t2i_top1: {:.2%}'.format(ac_top1_t2i))
        print('t2i_top5: {:.2%}'.format(ac_top5_t2i))
        print('t2i_top10: {:.2%}'.format(ac_top10_t2i))
        print('mAP: {:.2%}'.format(mAP))
        epochs.append(epoch)
        ac_top1_t2i = ac_top1_t2i.cuda().data.cpu().numpy()
        acc.append(ac_top1_t2i*100)
        loss.append(train_loss.cuda().data.cpu().numpy())

        plt.plot(epochs, acc, color='r', label='acc')  # r表示红色
        plt.plot(epochs, loss, color=(0, 0, 0), label='loss')  # 也可以用RGB值表示颜色
        plt.xlabel('epochs')  # x轴表示
        plt.ylabel('y label')  # y轴表示
        plt.title("chart")  # 图标标题表示
        plt.legend()  # 每条折线的label显示

        plt.show()  # 显示图片


if __name__=='__main__':
    args = parse_args()

    # load GPU
    str_ids = args.gpus.split(',')
    gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >= 0:
            gpu_ids.append(gid)

    # set gpu ids
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True  # make the training speed faster
    fix_seed(args.seed)

    name = args.name
    # set some paths
    checkpoint_dir = args.checkpoint_dir
    checkpoint_dir = os.path.join(checkpoint_dir, name)
    log_dir = args.log_dir
    log_dir = os.path.join(log_dir, name)

    sys.stdout = Logger(os.path.join(log_dir, "train_log.txt"))
    opt_dir = os.path.join('log', name)
    if not os.path.exists(opt_dir):
        os.makedirs(opt_dir)
    with open('%s/opts_train.yaml' % opt_dir, 'w') as fp:
        yaml.dump(vars(args), fp, default_flow_style=False)

    # pre-process the dataset
    transform_train_list = [
        transforms.Resize((args.height, args.width), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((args.height, args.width)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    transform_val_list = [
        transforms.Resize((args.height, args.width), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    # define dictionary: data_transforms
    data_transforms = {
        'train': transforms.Compose(transform_train_list),
        'val': transforms.Compose(transform_val_list),
    }

    dataloaders = {x: data_config(args.dir, args.batch_size, x, args.max_length, args.embedding_type, transform=data_transforms[x])
                   for x in ['train', 'val']}

    # loss function
    if args.CMPM:
        print("import CMPM")
    if args.CMPC:
        print("import CMPC")
    compute_loss = Loss(args).to(device)
    model = Network(args).to(device)

    # compute the model size:
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # load checkpoint:
    if args.resume is not None:
        start_epoch, model = load_checkpoint(model, args.resume)
    else:
        print("Do not load checkpoint,Epoch start from 0")
        start_epoch = 0

    # opitimizer:
    opitimizer = optimizer_function(args, model)
    exp_lr_scheduler = lr_scheduler(opitimizer, args)
    main(model, dataloaders, compute_loss, opitimizer, exp_lr_scheduler, start_epoch, args, checkpoint_dir)
    start_test()

