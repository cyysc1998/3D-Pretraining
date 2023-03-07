import os
import argparse
import yaml
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import ShapeNetPart

from models.model import ConsNet

from utils import obj_rotate_perm, obj_2_perm, emd_mixup, add_mixup
from utils import chamfer_distance, L1_loss, emd_loss, embed_cross_entropy, emd_L1_loss, centre_Loss
from utils import rand_proj, IOStream


seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]

def init(args, configpath):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
        os.system('cp ' + configpath + ' ./checkpoints/' + args.exp_name)
    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    return io


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(args, configpath):
    io = init(args, configpath)
    train_dataset = ShapeNetPart(partition='trainval', num_points=args.num_points)
    if (len(train_dataset) < 100):
        drop_last = False
    else:
        drop_last = True
    train_loader = DataLoader(train_dataset, num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=drop_last)
    test_loader = DataLoader(ShapeNetPart(partition='test', num_points=args.num_points), 
                            num_workers=8, batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    
    seg_num_all = train_loader.dataset.seg_num_all
    seg_start_index = train_loader.dataset.seg_start_index
    
    device = torch.device("cuda" if args.cuda else "cpu")
    
    if args.model == 'consnet':
        model = ConsNet(args, seg_num_all).to(device)
    elif args.model == 'pretrain':
        model = ConsNet(args, seg_num_all).to(device)
        model.load_state_dict(torch.load(args.pretrain_path))
    else:
        raise Exception("Not implemented")
        
    if args.parallel == True:
        model = nn.DataParallel(model)
        
    print(str(model))

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
        cur_lr = args.lr * 100
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        cur_lr = args.lr
        
    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
        print('Use COS')
    elif args.scheduler == 'coswarm':
        scheduler = CosineAnnealingWarmRestarts(opt, T_0=args.epochs//16, T_mult=2, eta_min=0.01*args.lr)
        args.epochs = args.epochs//16 * (1+2+4+8)
        print('Use Cos Warm')
        print('New epochs:', args.epochs)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=20, gamma=0.7)
        print('Use Step')


    io.cprint('Experiment: %s' % args.exp_name)

    if args.loss == 'l1loss':
        io.cprint('Use L1 Loss')
    elif args.loss == 'chamfer':
        io.cprint('Use Chamfer Distance')
    else:
        io.cprint('Not implemented')


    if args.l1loss:
        io.cprint('Add l1 loss')
    if args.l2loss:
        io.cprint('Add l2 loss')
    if args.embed_loss:
        io.cprint('Add embed_loss')
    if args.emd_l1loss:
        io.cprint('Add emd_l1_loss')

    
    iters = len(train_loader)

    # Train
    min_loss = 100
    io.cprint('Begin to train...')
    for epoch in range(args.epochs):
        io.cprint('=====================================Epoch %d========================================' % epoch)
        io.cprint('*****Train*****')
        # Train
        model.train()
        train_loss = 0
        for i, point in enumerate(train_loader):
            data, label, seg = point
            if epoch < 5:
                lr = 0.18 * cur_lr * epoch + 0.1 * cur_lr
                adjust_learning_rate(opt, lr)


            if args.task == '1obj_rotate':
                data1, data2, label1, label2 = obj_rotate_perm(data, label) # (B, N, 3)
            elif args.task == '2obj':
                data1, data2, label1, label2 = obj_2_perm(data, label) # (B, N, 3)
            elif args.task == 'alter':
                if epoch % 2 == 0:
                    data1, data2, label1, label2 = obj_rotate_perm(data, label) # (B, N, 3)
                else:
                    data1, data2, label1, label2 = obj_2_perm(data, label) # (B, N, 3)
            else:
                print('Task not implemented!')
                exit(0)
            
            if args.mixup == 'emd':
                mixup_data = emd_mixup(data1, data2) # (B, N, 3)
            elif args.mixup == 'add':
                mixup_data = add_mixup(data1, data2) # (B, N, 3)

                
            mixup_data = mixup_data.permute(0, 2, 1) # (B, 3, N)
            batch_size = mixup_data.size()[0]
            
            seg = seg - seg_start_index


            if args.use_one_hot:
                label_one_hot1 = np.zeros((batch_size, 16))
                label_one_hot2 = np.zeros((batch_size, 16))
                for idx in range(batch_size):
                    label_one_hot1[idx, label1[idx]] = 1
                    label_one_hot2[idx, label2[idx]] = 1
                
                label_one_hot1 = torch.from_numpy(label_one_hot1.astype(np.float32))
                label_one_hot2 = torch.from_numpy(label_one_hot2.astype(np.float32))
            else:
                label_one_hot1 = torch.rand(batch_size, 16)
                label_one_hot2 = torch.rand(batch_size, 16)
                
            data, label_one_hot1, label_one_hot2, seg = data.to(device), label_one_hot1.to(device), label_one_hot2.to(device), seg.to(device)

            # Project
            use_xyz = (args.extra_len == 3)
            proj1 = rand_proj(data1, use_xyz)
            proj2 = rand_proj(data2, use_xyz)
            
            # Train
            opt.zero_grad()
            
            pred1, embed1, logit_scale1 = model(mixup_data, proj1, label_one_hot1)
            pred2, embed2, logit_scale2 = model(mixup_data, proj2, label_one_hot2)

            pred1 = pred1.permute(0, 2, 1) # (B, N, 3)
            pred2 = pred2.permute(0, 2, 1) # (B, N, 3)

            if args.loss == 'l1loss':
                loss = L1_loss(pred1, data1) + L1_loss(pred2, data2)
            elif args.loss == 'chamfer':
                loss1 = chamfer_distance(pred1, data1) + chamfer_distance(data1, pred1)
                loss2 = chamfer_distance(pred2, data2) + chamfer_distance(data2, pred2)
                loss = loss1 + loss2
            elif args.loss == 'emd':
                loss = emd_loss(pred1, data1) + emd_loss(pred2, data2)

            
            # loss1 = chamfer_distance(pred1, data1) + chamfer_distance(data1, pred1)
            # loss2 = chamfer_distance(pred2, data2) + chamfer_distance(data2, pred2)
            # loss_emd = emd_loss(pred1, data1) + emd_loss(pred2, data2)
            # loss = loss1 + loss2 + 0.1 * loss_emd
                

            if args.l2loss:
                l2_loss = l2_loss + nn.MSELoss()(pred1, data1) + nn.MSELoss()(pred2, data2)
                loss = loss + args.l2_param * l2_loss

            if args.l1loss:
                L1_loss = nn.L1Loss()
                l1_loss = L1_loss(pred1, data1) + L1_loss(pred2, data1)
                loss = loss + args.l1_param * l1_loss

            if args.embed_loss:
                loss = loss + embed_cross_entropy(embed1, logit_scale1)

            if args.emd_l1loss:
                emd_l1loss = emd_L1_loss(pred1, data1) + emd_L1_loss(pred2, data2)
                loss = loss + args.emd_l1loss_param + emd_l1loss 

            if hasattr(args, 'centreloss'):
                if args.centreloss:
                    eps = args.centreloss_eps
                    centre_loss =  centre_Loss(pred1, eps) + centre_Loss(pred2, eps)
                    loss = loss + args.centre_param * centre_loss

            loss.backward()
            
            train_loss = train_loss + loss.item()
            opt.step()

            if (i + 1) % 100 == 0:
                io.cprint('iters %d, tarin loss: %.6f' % (i, train_loss / i))

        io.cprint('Learning rate: %.6f' % (opt.param_groups[0]['lr']))

        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'coswarm':
            scheduler.step(epoch + i / iters)
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5

        # Test 
        if args.valid:
            io.cprint('*****Test*****')

            test_loss = 0
            l1_loss_count = 0
            l2_loss_count = 0
            embed_loss_count = 0
            emd_l1_loss_count = 0
            
            model.eval()
            for data, label, seg in test_loader:
                with torch.no_grad():
                    if args.task == '1obj_rotate':
                        data1, data2, label1, label2 = obj_rotate_perm(data, label) # (B, N, 3)
                    elif args.task == '2obj':
                        data1, data2, label1, label2 = obj_2_perm(data, label) # (B, N, 3)
                    elif args.task == 'alter':
                        if epoch % 2 == 0:
                            data1, data2, label1, label2 = obj_rotate_perm(data, label) # (B, N, 3)
                        else:
                            data1, data2, label1, label2 = obj_2_perm(data, label) # (B, N, 3)
                    else:
                        print('Task not implemented!')
                        exit(0)
                    
                    if args.mixup == 'emd':
                        mixup_data = emd_mixup(data1, data2) # (B, N, 3)
                    elif args.mixup == 'add':
                        mixup_data = add_mixup(data1, data2) # (B, N, 3)
                            
                    mixup_data = mixup_data.permute(0, 2, 1) # (B, 3, N)
                    batch_size = mixup_data.size()[0]

                    seg = seg - seg_start_index
                    label_one_hot1 = np.zeros((batch_size, 16))
                    label_one_hot2 = np.zeros((batch_size, 16))
                    for idx in range(batch_size):
                        label_one_hot1[idx, label1[idx]] = 1
                        label_one_hot2[idx, label2[idx]] = 1

                    label_one_hot1 = torch.from_numpy(label_one_hot1.astype(np.float32))
                    label_one_hot2 = torch.from_numpy(label_one_hot2.astype(np.float32))
                    data, label_one_hot1, label_one_hot2, seg = data.to(device), label_one_hot1.to(device), label_one_hot2.to(device), seg.to(device)


                    use_xyz = (args.extra_len == 3)
                    proj1 = rand_proj(data1, use_xyz)
                    proj2 = rand_proj(data2, use_xyz)
                    
                    pred1, embed1, logit_scale1 = model(mixup_data, proj1, label_one_hot1)
                    pred2, embed2, logit_scale2 = model(mixup_data, proj2, label_one_hot2)

                    pred1 = pred1.permute(0, 2, 1) # (B, N, 3)
                    pred2 = pred2.permute(0, 2, 1) # (B, N, 3)


                    if args.loss == 'l1loss':
                        loss = L1_loss(pred1, data1) + L1_loss(pred2, data2)
                    elif args.loss == 'chamfer':
                        loss1 = chamfer_distance(pred1, data1) + chamfer_distance(data1, pred1)
                        loss2 = chamfer_distance(pred2, data2) + chamfer_distance(data2, pred2)
                        loss = loss1 + loss2
                    elif args.loss == 'emd':
                        loss = emd_loss(pred1, data1) + emd_loss(pred2, data2)


                    if args.l2loss:
                        l2_loss = nn.MSELoss()(pred1, data1) + nn.MSELoss()(pred2, data2)
                        l2_loss_count = l2_loss_count + l2_loss
                        loss = loss + args.l2_param * l2_loss

                    if args.l1loss:
                        L1_loss = nn.L1Loss()
                        l1_loss = L1_loss(pred1, data1) + L1_loss(pred2, data1)
                        l1_loss_count = l1_loss_count + l1_loss
                        loss = loss + args.l1_param * l1_loss

                    if args.embed_loss:
                        embed_loss = embed_cross_entropy(embed1, logit_scale1)
                        embed_loss_count = embed_loss_count + embed_loss
                        loss = loss + embed_loss

                    if args.emd_l1loss:
                        emd_l1loss = emd_L1_loss(pred1, data1) + emd_L1_loss(pred2, data2)
                        emd_l1_loss_count = emd_l1_loss_count + emd_l1loss
                        loss = loss + args.emd_l1loss_param * emd_l1loss 


                    test_loss = test_loss + loss.item()

            if args.l1loss:
                io.cprint('L1 loss:%.6f' % (l1_loss_count / len(test_loader)))
            if args.l2loss:
                io.cprint('L2 loss:%.6f' % (l2_loss_count / len(test_loader)))
            if args.embed_loss:
                io.cprint('Embed loss:%.6f' % (embed_loss_count / len(test_loader)))
            if args.emd_l1loss:
                io.cprint('Emd L1 loss:%.6f' % (emd_l1_loss_count / len(test_loader)))
            
            io.cprint('Train loss: %.6f, Test loss: %.6f' % (train_loss / len(train_loader), test_loss / len(test_loader)))
            cur_loss = test_loss / len(test_loader)
            if cur_loss < min_loss:
                min_loss = cur_loss
                if args.parallel == True:
                    torch.save(model.module.state_dict(), 'checkpoints/%s/best_%s.pkl' % (args.exp_name, args.exp_name))
                else:
                    torch.save(model.state_dict(), 'checkpoints/%s/best_%s.pkl' % (args.exp_name, args.exp_name))
        if args.parallel == True:
            if (epoch + 1) % 10 == 0:            
                torch.save(model.module.state_dict(), 'checkpoints/%s/%s_epoch_%s.pkl' % (args.exp_name, args.exp_name, str(epoch)))   
        else:
            if (epoch + 1) % 10 == 0:            
                torch.save(model.state_dict(), 'checkpoints/%s/%s_epoch_%s.pkl' % (args.exp_name, args.exp_name, str(epoch))) 
    torch.save(model.state_dict(), 'checkpoints/%s/%s.pkl' % (args.exp_name, args.exp_name))
    


def test():
    pass

class Config(dict):
    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
        self.__dict__ = self



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point Cloud')
    parser.add_argument('--opt', type=str, default='train.yml', metavar='N',
                        help='config yaml')  
    args = parser.parse_args()
    configyaml = args.opt
    configpath = str(args.opt)

    with open(configyaml, 'r', encoding='utf-8') as f:
        config = yaml.safe_load_all(f)
        config = list(config)[0]
    
    config = Config(config)
    
    if config.eval:
        test(config, configpath)
    else:
        train(config, configpath)
