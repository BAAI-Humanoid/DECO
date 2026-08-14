import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from torch.amp import autocast


def train(net, net_without_ddp, train_loader, optimizer, warmup_scheduler, lr_scheduler, epoch, opt, scaler, local_rank, rank):
    if rank == 0:
        print("Start training")
        pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{opt.epochs}', postfix=dict, mininterval=0.3)
    net.train()
    total_loss = 0

    for batch_idx, (imgs, obs_state, action, mask, prompt, prompt_mask) in enumerate(train_loader):
        imgs = imgs.cuda(local_rank)  # (b, n_view, 3, h, w)
        obs_state = obs_state.cuda(local_rank)    # (b, 28)
        action = action.cuda(local_rank)  # (b, chunksize, 28)
        mask = mask.cuda(local_rank)  # (b, chunksize)
        mask = mask.unsqueeze(-1).repeat(1, 1, action.shape[-1])  # (b, chunksize, 28)
        prompt = prompt.cuda(local_rank)  # (b, prompt_len, 768)
        prompt_mask = prompt_mask.cuda(local_rank)  # (b, prompt_len)
        
        optimizer.zero_grad()
        global_step = (epoch - 1) * len(train_loader) + batch_idx  # current global training step
        if global_step < opt.warm_up_steps:
            warmup_scheduler.step()
        else:
            lr_scheduler.step()  # cosine LR decays per step (called once per training step after warmup)

        with autocast(device_type='cuda', enabled=opt.amp, dtype=torch.bfloat16):
            out, noise = net(imgs, obs=obs_state, act=action, prompt=prompt, prompt_mask=prompt_mask, training=True)
            # loss = (F.mse_loss(out, noise - action, reduction='none') * mask.unsqueeze(-1)).sum() / mask.sum()
            loss = F.mse_loss(out, noise - action)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

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
                   os.path.join(opt.logs, 'epoch_%d_loss_%f.pth') % (epoch, total_loss))  # save model

    if rank == 0:
        with open(os.path.join(opt.logs, 'result.txt'), 'a+') as f:
            f.writelines('\nEpoch: %d, total loss: %f, epoch loss: %f' % (epoch, total_loss, epoch_loss))

    return epoch_loss


def val(net, test_loader, epoch, opt, act_dim, chunksize, local_rank, rank):
    if rank == 0:
        print("Start validation")
        pbar = tqdm(total=len(test_loader), desc=f'Epoch {epoch}/{opt.epochs}', postfix=dict, mininterval=0.3)
    net.eval()
    total_loss = 0
    mae = torch.zeros(chunksize, act_dim).cuda(local_rank)  # (chunksize, act_dim) for tracking the model's L1 error
    
    with torch.no_grad():
        for batch_idx, (imgs, obs, action, mask, prompt, prompt_mask) in enumerate(test_loader):
            imgs = imgs.cuda(local_rank)
            obs = obs.cuda(local_rank)
            action = action.cuda(local_rank)
            mask = mask.cuda(local_rank)
            prompt = prompt.cuda(local_rank)
            prompt_mask = prompt_mask.cuda(local_rank)
            with autocast(device_type='cuda', enabled=opt.amp, dtype=torch.bfloat16):
                out = net(imgs, obs=obs, act=action, prompt=prompt, prompt_mask=prompt_mask, training=False)

            mask = mask.unsqueeze(-1).repeat(1, 1, act_dim)
            loss = (mask * F.l1_loss(out, action, reduction='none')).sum() / mask.sum()
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
    mae = torch.round(mae * 100) / 100 # round to two decimals when printing
    epoch_loss = total_loss / len(test_loader)
    if rank == 0:
        print("\nVal epoch loss: %f" % epoch_loss)
        print('\nmae: ', mae)
        with open(os.path.join(opt.logs, 'result.txt'), 'a+') as f:
            f.writelines('\nVal epoch: %d, total loss: %f, epoch loss: %f \n MAE: %s \n\n' % (epoch, total_loss, epoch_loss, str(mae)))
    return epoch_loss


