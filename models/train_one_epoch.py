import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from torch.amp import autocast


def train(net, net_without_ddp, train_loader, optimizer, criterion, warmup_scheduler, epoch, opt, act_dim, obs_state, scaler, local_rank, rank):
    if rank == 0:
        print("Start training")
        pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{opt.epochs}', postfix=dict, mininterval=0.3)
    net.train()
    total_loss = 0

    for batch_idx, (imgs, tacs, obs, action, mask, task_idx) in enumerate(train_loader):
        imgs = imgs.cuda(local_rank)  # (b, n_view, 3, h, w)
        obs = obs.cuda(local_rank)    # (b, 28)
        action = action.cuda(local_rank)  # (b, chunksize, 28)
        task_idx = task_idx.cuda(local_rank)  # (b, )
        tacs = tacs.cuda(local_rank) if tacs is not None else None
        mask = mask.cuda(local_rank)
        
        # NOTE: mask is not used in diffusion loss
        optimizer.zero_grad()
        if epoch <= opt.warm_up_epoch:
            warmup_scheduler.step()
        if not opt.amp:
            out, noise = net(imgs, obs=obs, act=action, task_idx=task_idx, tacs=tacs, training=True)
            if mask.ndim == 3:
                loss = F.mse_loss(out, noise - action, reduction='none')
                loss = (loss * mask).sum() / mask.sum()
            else:
                loss = F.mse_loss(out, noise - action)
            loss.backward()
            optimizer.step()
        else:
            with autocast(device_type='cuda', enabled=True, dtype=torch.float16):
                out, noise = net(imgs, obs=obs, act=action, task_idx=task_idx, tacs=tacs, training=True) # (b, chunksize, 26)
                if mask.ndim == 3:
                    loss = F.mse_loss(out, noise - action, reduction='none')
                    loss = (loss * mask).sum() / mask.sum()
                else:
                    loss = F.mse_loss(out, noise - action)  
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total_loss += loss.item()

        if rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (batch_idx + 1),
                                'lr': optimizer.state_dict()['param_groups'][0]['lr']})
            pbar.update(1)
            with open(os.path.join(opt.logs, 'result.txt'), 'a+') as f:
                f.writelines("Epoch:%d [%d|%d] loss:%f \n" % (epoch, batch_idx + 1, len(train_loader), loss.mean()))
                
    if dist.is_initialized():
        dist.barrier()
    epoch_loss = total_loss / len(train_loader)
    if epoch % opt.save_period == 0 and rank == 0:
        print('save model to logs')
        torch.save(net_without_ddp.state_dict(),
                   os.path.join(opt.logs, 'epoch_%d_loss_%f.pth') % (epoch, total_loss))  # 保存模型

    if rank == 0:
        with open(os.path.join(opt.logs, 'result.txt'), 'a+') as f:
            f.writelines('\nEpoch: %d, total loss: %f, epoch loss: %f' % (epoch, total_loss, epoch_loss))

    return epoch_loss


def val(net, test_loader, criterion, epoch, opt, act_dim, chunksize, obs_state, local_rank, rank):
    if rank == 0:
        print("Start validation")
        pbar = tqdm(total=len(test_loader), desc=f'Epoch {epoch}/{opt.epochs}', postfix=dict, mininterval=0.3)
    net.eval()
    total_loss = 0
    mae = torch.zeros(chunksize, act_dim).cuda(local_rank)  # (chunksize, 26) 用于统计模型的l1误差
    
    with torch.no_grad():
        for batch_idx, (imgs, tacs, obs, action, mask, task_idx) in enumerate(test_loader):
            imgs = imgs.cuda(local_rank)

            obs = obs.cuda(local_rank)
            action = action.cuda(local_rank)
            mask = mask.cuda(local_rank)
            task_idx = task_idx.cuda(local_rank)
            tacs = tacs.cuda(local_rank) if tacs is not None else None

            out = net(imgs, obs=obs, act=action, task_idx=task_idx, tacs=tacs, training=False)
            if mask.ndim == 2:
                mask = mask.unsqueeze(-1).repeat(1, 1, act_dim)
            loss = (mask * criterion(out, action)).sum() / mask.sum()
            total_loss += loss.item()
            
            ae = (mask * torch.abs(out - action)).sum(0) # (chunksize, 28)
            if dist.is_initialized():
                dist.all_reduce(ae)
            mae += ae
            if rank == 0:
                ae = [round(x / opt.batch_size / chunksize, 2) for x in ae.sum(0).tolist()]
                pbar.set_postfix(**{'val_loss': total_loss / (batch_idx + 1), 'AE': ae})
                pbar.update(1)
                
    if dist.is_initialized():
        dist.barrier()
    mae = mae / len(test_loader) / opt.batch_size
    mae = torch.round(mae * 100) / 100 # 打印的时候保留两位
    epoch_loss = total_loss / len(test_loader)
    if rank == 0:
        print("\nVal epoch loss: %f" % epoch_loss)
        print('\nmae: ', mae)
        with open(os.path.join(opt.logs, 'result.txt'), 'a+') as f:
            f.writelines('\nVal epoch: %d, total loss: %f, epoch loss: %f \n MAE: %s \n\n' % (epoch, total_loss, epoch_loss, str(mae)))
    return epoch_loss


