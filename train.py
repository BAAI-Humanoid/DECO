import os
import yaml
import time
import random
import scipy
import argparse
import importlib
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from dataset import my_Dataset, letterbox
from torchvision.transforms import v2 as transforms
try:
    from torch.optim.lr_scheduler import _LRScheduler
except:
    from torch.optim.lr_scheduler import LRScheduler as _LRScheduler
import warnings
warnings.filterwarnings("ignore")


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler

    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


class LossHistory():
    def __init__(self, log_dir, val_loss_flag=True):
        self.log_dir = log_dir
        self.val_loss_flag = val_loss_flag

        self.losses = []
        if self.val_loss_flag:
            self.val_loss = []

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)


    def append_loss(self, loss, val_loss=None):
        self.losses.append(loss)
        if self.val_loss_flag:
            self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "train_loss.txt"), 'a') as f:
            f.write(f"{loss}\n")
            
        if self.val_loss_flag:
            with open(os.path.join(self.log_dir, "val_loss.txt"), 'a') as f:
                f.write(f"{val_loss}\n")

        self.plot_train_loss()
        if self.val_loss_flag:
            self.plot_val_loss()

    def plot_train_loss(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        try:
            if len(self.losses) >= 5:
                window = 5 if len(self.losses) < 25 else 15
                smooth = scipy.signal.savgol_filter(self.losses, window, 3)
                plt.plot(iters, smooth, 'green', linestyle='--', linewidth=2, label='smooth train loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Train Loss')
        plt.legend()
        plt.savefig(os.path.join(self.log_dir, "train_loss.png"))
        plt.cla()
        plt.close("all")

    def plot_val_loss(self):
        iters = range(len(self.val_loss))
        plt.figure()
        plt.plot(iters, self.val_loss, 'blue', linewidth=2, label='val loss')
        try:
            if len(self.val_loss) >= 5:
                window = 5 if len(self.val_loss) < 25 else 15
                smooth = scipy.signal.savgol_filter(self.val_loss, window, 3)
                plt.plot(iters, smooth, 'purple', linestyle='--', linewidth=2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Val Loss')
        plt.legend()
        plt.savefig(os.path.join(self.log_dir, "val_loss.png"))
        plt.cla()
        plt.close("all")


def main(opt):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.device_id
    Cuda = True if torch.cuda.is_available() else False
    ngpus_per_node = torch.cuda.device_count()
    old_dir = opt.logs
    if opt.distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        rank = int(os.environ["RANK"])
        dist.init_process_group(backend="nccl")
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("GPU Device Count : ", ngpus_per_node)
    else:
        local_rank = 0

    if local_rank == 0:
        if os.path.exists(opt.logs):
            print('log dir exists, change to a new dir')
            opt.logs = opt.logs + str(random.randint(0, 1000))
        print('create log dir:', opt.logs)
        loss_dir = os.path.join(opt.logs, 'loss')
        if not os.path.exists(loss_dir):
            os.makedirs(loss_dir)
        loss_history = LossHistory(loss_dir)
        with open(os.path.join(opt.logs, 'result.txt'), 'a+') as f:
            f.writelines('----------------training logs----------------\n %s \n \n' % str(opt))

    else:
        loss_history = None


    config = yaml.safe_load(open(opt.config, 'r'))
    act_dim, chunksize, obs_state = config['model']['action_dim'], config['model']['chunk_size'], config['model']['obs_state']
    img_size = config['img']['img_size']
    img_mean, img_std = config['img']['img_mean'], config['img']['img_std']
    model_name = config['model_name']
    find_unused_parameters = model_name == 'deco'
    importmodule = importlib.import_module(f"models.{model_name}")

    if img_size[0] != img_size[1]:
        print('use transform.Resize to ', img_size)
        resize_transform = transforms.Resize(img_size)
    else:
        print('use letterbox to ', img_size)
        resize_transform = letterbox(img_size[0], fill=128)
        
    train_transform = transforms.Compose([
        resize_transform,
        transforms.RandomApply([transforms.ColorJitter(brightness=(0.7, 1.3), contrast=(0.8, 1.2), saturation=(0.8, 1.2))], p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(random.choice([3, 5, 7])), sigma=random.uniform(0.1, 2))], p=0.5),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(
            mean=img_mean, 
            std=img_std),
        ])

    test_transform = transforms.Compose([
        resize_transform,
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(
            mean=img_mean, 
            std=img_std)
        ])
    train_dataset = my_Dataset(data_dir=opt.data, train=True, transform=train_transform, **config['data'])
    test_dataset = my_Dataset(data_dir=opt.data, train=False, transform=test_transform, **config['data'])

    if opt.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        batch_size = opt.batch_size // ngpus_per_node
        shuffle = False
    else:
        batch_size = opt.batch_size
        train_sampler, val_sampler = None, None
        shuffle = True

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                                               num_workers=opt.num_workers, pin_memory=True, drop_last=True,
                                               sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle,
                                              num_workers=opt.num_workers, pin_memory=True, drop_last=True,
                                              sampler=val_sampler)

    net = importmodule.modeling(**config['model']) 
        
    if Cuda:
        if opt.distributed:  # DDP
            net = net.cuda(local_rank)
            for param in net.parameters():  
                dist.broadcast(param.data, src=0)
            net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank,
                                                            find_unused_parameters=find_unused_parameters)
            net_without_ddp = net.module
            cudnn.benchmark = True
        else:
            if torch.cuda.device_count() > 1:  # DP
                net = torch.nn.DataParallel(net)
                cudnn.benchmark = True
                net = net.cuda()
                net_without_ddp = net.module
            else:  # Single GPU
                cudnn.benchmark = True
                net = net.cuda()
                net_without_ddp = net
    else:
        print('Please try to use GPU, otherwise it will take long time for training')
        net_without_ddp = net

    if opt.amp:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    if not opt.adamw:
        if local_rank == 0:
            print('use sgd')
        optimizer = torch.optim.SGD(params=(p for p in net_without_ddp.parameters() if p.requires_grad), lr=opt.lr, momentum=0.843, weight_decay=0.00036)
    else:
        if local_rank == 0:
            print('use adamw')
        optimizer = torch.optim.AdamW(params=(p for p in net_without_ddp.parameters() if p.requires_grad), lr=opt.lr, betas=(0.95, 0.999), weight_decay=1e-6)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=opt.lr_f)
    warmup_scheduler = WarmUpLR(optimizer, len(train_loader) * opt.warm_up_epoch)
    criterion = torch.nn.L1Loss(reduction='none')

    if opt.resume:
        if local_rank == 0:
            print('resume from last weights')
        checkpoint = torch.load(os.path.join(old_dir, "last_weights.pth"), map_location='cpu')  # 读取之前保存的权重文件(包括优化器以及学习率策略)
        net_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        if opt.amp:
            scaler.load_state_dict(checkpoint["scaler"])
    else:
        start_epoch = 1

    loss = 1000
    val_loss = 1000
    save_epoch = 0
    for epoch in range(start_epoch, opt.epochs + 1):
        if opt.distributed:
            train_sampler.set_epoch(epoch)
        train_loss = importmodule.train(net, net_without_ddp, train_loader, optimizer, criterion, warmup_scheduler, epoch, opt, act_dim, obs_state, scaler, local_rank)
        if epoch % opt.val_per_epoch == 0:
            val_loss = importmodule.val(net, test_loader, criterion, epoch, opt, act_dim, chunksize, obs_state, local_rank)
        if val_loss < loss:
            loss = val_loss
            save_epoch = epoch
            if local_rank == 0:
                print('save best model to logs!')
                torch.save(net_without_ddp.state_dict(), os.path.join(opt.logs, 'best.pth'))
                
        if epoch > opt.warm_up_epoch:
            lr_scheduler.step()
            
        if local_rank == 0:
            loss_history.append_loss(train_loss, val_loss)
            print('current best epoch:', save_epoch, '\n')
            print('lr:', optimizer.state_dict()['param_groups'][0]['lr'], '\n')
            save_file = {'model': net_without_ddp.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'lr_scheduler': lr_scheduler.state_dict(),
                         'epoch': epoch}
            if opt.amp:
                save_file["scaler"] = scaler.state_dict()
            torch.save(save_file, os.path.join(opt.logs, "last_weights.pth"))

    if dist.is_initialized():
        dist.destroy_process_group()
        
if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/deco.yaml', help='Path to configuration YAML file')
    parser.add_argument('--data', type=str, default='./data', help='Path to dataset directory')
    parser.add_argument('--adamw', default=True, type=bool, help='Use AdamW optimizer instead of SGD')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--lr_f', type=float, default=5e-6, help='Final learning rate (minimum for cosine annealing)')
    parser.add_argument('--batch-size', type=int, default=128*8, help='Total batch size across all GPUs')
    parser.add_argument('--warm_up_epoch', type=int, default=1, help='Number of warmup epochs for learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='Total number of training epochs')
    parser.add_argument('--val_per_epoch', type=int, default=1, help='Number of epochs between validations')
    parser.add_argument('--logs', type=str, default='./logs/log_dp', help='Directory to save logs and models')
    parser.add_argument('--save_period', type=int, default=20, help='Number of epochs between model checkpoints')
    parser.add_argument('--resume', action='store_true', help='Resume training from the most recent checkpoint')
    parser.add_argument('--local_rank', default=-1, type=int, help='Local rank for distributed training (auto-set by torch.distributed)')
    parser.add_argument('--distributed', default=True, type=bool, help='Enable Distributed Data Parallel (DDP) training')
    parser.add_argument('--amp', default=True, type=bool, help='Enable automatic mixed precision training')
    parser.add_argument('--num-workers', type=int, default=16, help='Number of workers for data loading')
    parser.add_argument('--device_id', default='0, 1, 2, 3, 4, 5, 6, 7', type=str, help='Comma-separated list of CUDA device IDs to use')

    opt = parser.parse_args()
    main(opt)
    end_time = time.time()
    run_time_in_seconds = end_time - start_time

    hours, remainder = divmod(run_time_in_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"Total training time: {int(hours)} h {int(minutes)} m {int(seconds)} s")