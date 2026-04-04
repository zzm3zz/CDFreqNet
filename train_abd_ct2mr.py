import os
from datetime import datetime
import torch
from torch import nn
from models.network import CDFreqNet
from utils.STN import SpatialTransformer
from utils.Transform_self import SpatialTransform
from utils.dataloader import Dataset3D_DFI as TrainDataset
from utils.dataloader import Dataset3D  
from torch.utils.data import DataLoader
from utils.losses import dice_loss, prob_entropyloss
from utils.utils import AverageMeter, LogWriter, dice
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils.DynamicTemporalConstraint import DynamicTemporalConstraint, SpatialWeighted_DiceLoss

def crt_file(path):
    os.makedirs(path, exist_ok=True)

class Trainer(object):
    def __init__(self, args=None):
        super(Trainer, self).__init__()

        self.start_epoch = args.start_epoch
        self.epoches = args.num_epoch
        self.iters = args.num_iters
        self.save_epoch = args.save_epoch

        self.model_name = args.model_name
        self.direction = args.direction
        self.part = args.part

        self.lr_seg = args.lr_seg
        self.bs = args.batch_size
        self.n_classes = args.num_classes
        self.srs_rmmax = args.srs_rmmax
        self.tar_rmmax = args.tar_rmmax

        A_root = args.A_root
        B_root = args.B_root
        Val_root = args.Val_root

        data_listA = [os.path.join(A_root, k) for k in os.listdir(A_root) if not k.startswith('.')]
        data_listB = [os.path.join(B_root, k) for k in os.listdir(B_root) if not k.startswith('.')]
        
        # Validation path logic
        val_path = Val_root
        if os.path.exists(val_path):
            data_list_val = [os.path.join(val_path, k) for k in os.listdir(val_path) if not k.startswith('.')]
        else:
            print(f"Warning: Validation path {val_path} not found. Using training B data for validation.")
            data_list_val = data_listB

        train_srs = data_listA # Source Labeled
        train_tar = data_listB # Target Unlabeled
        val_data = data_list_val

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")  

        self.checkpoint_dir = os.path.join(args.checkpoint_root, self.model_name + '_' + timestamp + '_' + str(self.direction) + '_' + str(self.part))
        crt_file(self.checkpoint_dir)
        crt_file(self.checkpoint_dir)

        # Data augmentation (Source & Target Spatial Augmentation)
        self.spatial_aug = SpatialTransform(do_rotation=True,
                                            angle_x=(-np.pi / 9, np.pi / 9),
                                            angle_y=(-np.pi / 9, np.pi / 9),
                                            angle_z=(-np.pi / 9, np.pi / 9),
                                            do_scale=True,
                                            scale_x=(0.75, 1.25),
                                            scale_y=(0.75, 1.25),
                                            scale_z=(0.75, 1.25),
                                            do_translate=True,
                                            trans_x=(-0.1, 0.1),
                                            trans_y=(-0.1, 0.1),
                                            trans_z=(-0.1, 0.1),
                                            do_shear=True,
                                            shear_xy=(-np.pi / 18, np.pi / 18),
                                            shear_xz=(-np.pi / 18, np.pi / 18),
                                            shear_yx=(-np.pi / 18, np.pi / 18),
                                            shear_yz=(-np.pi / 18, np.pi / 18),
                                            shear_zx=(-np.pi / 18, np.pi / 18),
                                            shear_zy=(-np.pi / 18, np.pi / 18),
                                            do_elastic_deform=True,
                                            alpha=(0., 512.),
                                            sigma=(10., 13.))

        # initialize model
        self.model = CDFreqNet(input_channels=1, num_classes=self.n_classes).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_seg, weight_decay=1e-4)
        self.stn = SpatialTransformer()

        trainsrs_dataset = TrainDataset(train_srs, rmmax=self.srs_rmmax)
        traintar_dataset = TrainDataset(train_tar, rmmax=self.tar_rmmax)
     
        self.dataloader_srstrain = DataLoader(trainsrs_dataset, batch_size=self.bs, shuffle=True, drop_last=True, num_workers=1, pin_memory=True)
        self.dataloader_tartrain = DataLoader(traintar_dataset, batch_size=self.bs, shuffle=True, drop_last=True, num_workers=1, pin_memory=True)
        
        # Validation dataloader
        val_dataset = Dataset3D(val_data)
   
        self.dataloader_val = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

        # define loss
        self.L_seg = dice_loss
        self.L_mse = nn.MSELoss() 
        
        self.dtc = DynamicTemporalConstraint(num_classes=self.n_classes, 
                                      tau=0.5, feat_channels=64).cuda()

        self.criterion_seg = SpatialWeighted_DiceLoss(num_classes=self.n_classes).cuda()
        self.criterion_cons = SpatialWeighted_DiceLoss(num_classes=self.n_classes).cuda()
        
        # define loss log
        self.L_seg_log = AverageMeter(name='L_Seg')
        self.L_consist_log = AverageMeter(name='L_consist')
        self.L_ent_log = AverageMeter(name='L_ent') 
        
        # Define validation logs
        self.L_val_dice_log = AverageMeter(name='Val_Dice')
        self.L_val_loss_log = AverageMeter(name='Val_Loss')

        self.history = {
            'epoch': [],
            'train_seg_loss': [],
            'val_dice': []
        }

    def to_categorical(self, y, num_classes=None):
        y = np.array(y, dtype='int')
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((num_classes, n))
        categorical[y, np.arange(n)] = 1
        output_shape = (num_classes,) + input_shape
        categorical = np.reshape(categorical, output_shape)
        return categorical

    def train_iterator(self, srs_high, srs_low, srs_high_r, srs_low_r, srs_label, 
                       tar_high, tar_low, tar_high_r, tar_low_r, epoch, iters):

        self.optimizer.zero_grad()

        # ============================================================
        # Part 1: Source Domain Training (Supervised + Consistency)
        # ============================================================
        pred_src_factual, feat_src_factual = self.model(x_high=srs_high, x_low=srs_low, 
                                    rmmax=self.srs_rmmax)
        pred_src_intervention, feat_src_intervention = self.model(x_high=srs_high_r, x_low=srs_low_r, 
                                    rmmax=self.srs_rmmax)

        pred_tar_factual, feat_tar_factual = self.model(x_high=tar_high, x_low=tar_low, 
                                    rmmax=self.tar_rmmax)
        pred_tar_intervention, feat_tar_intervention = self.model(x_high=tar_high_r, x_low=tar_low_r, 
                                    rmmax=self.tar_rmmax)
        
        w_src, w_tgt, loss_align = self.dtc(
            f_factual_src=feat_src_factual, 
            f_intervention_src=feat_src_intervention, 
            p_intervention_src=pred_src_intervention,    
            mask_src=srs_label,
            f_factual_tgt=feat_tar_factual, 
            f_intervention_tgt=feat_tar_intervention, 
            p_intervention_tgt=pred_tar_intervention, 
            mask_tgt=pred_tar_factual.detach(),
            current_epoch=epoch
        )
       
        loss_seg_src_factual = self.criterion_seg(pred_src_factual, srs_label, weight_map=None)
                           
        loss_seg_src = self.criterion_seg(pred_src_intervention, srs_label, weight_map=w_src)
                           
        loss_cons_tar = self.criterion_seg(pred_tar_factual, pred_tar_intervention, weight_map=w_tgt) * 0.5

        w_src_seg = 1.0
        w_cons = 1.0

        loss_total = w_src_seg * (loss_seg_src + loss_seg_src_factual) + \
                     w_cons * (loss_cons_tar)
        loss_total.backward()
        self.optimizer.step()

        self.L_seg_log.update(loss_seg_src.item(), srs_high.size(0))
        self.L_consist_log.update(loss_cons_tar.item() if torch.is_tensor(loss_cons_tar) else loss_cons_tar, srs_high.size(0))

    def train_epoch(self, epoch):
        self.model.train()

        loader_src = iter(self.dataloader_srstrain)
        loader_tar = iter(self.dataloader_tartrain)

        for i in range(self.iters):
            try:
                srs_high, srs_low, srs_high_r, srs_low_r, srslabel = next(loader_src)
            except StopIteration:
                loader_src = iter(self.dataloader_srstrain)
                srs_high, srs_low, srs_high_r, srs_low_r, srslabel = next(loader_src)
            try:
                tar_high, tar_low, tar_high_r, tar_low_r, _ = next(loader_tar)
            except StopIteration:
                loader_tar = iter(self.dataloader_tartrain)
                tar_high, tar_low, tar_high_r, tar_low_r, _ = next(loader_tar)

            if torch.cuda.is_available():
                srs_high = srs_high.cuda()
                srs_low = srs_low.cuda()
                srs_high_r = srs_high_r.cuda()
                srs_low_r = srs_low_r.cuda()
                srslabel = srslabel.cuda()
                
                tar_high = tar_high.cuda()
                tar_low = tar_low.cuda()
                tar_high_r = tar_high_r.cuda()
                tar_low_r = tar_low_r.cuda()
                
            mat, code_spa = self.spatial_aug.rand_coords(srs_high.shape[2:])
            
            srs_high = self.spatial_aug.augment_spatial(srs_high, mat, code_spa)
            srs_low = self.spatial_aug.augment_spatial(srs_low, mat, code_spa)
            srs_high_r = self.spatial_aug.augment_spatial(srs_high_r, mat, code_spa)
            srs_low_r = self.spatial_aug.augment_spatial(srs_low_r, mat, code_spa)
            
            srslabel = self.spatial_aug.augment_spatial(srslabel, mat, code_spa, mode="nearest").int()
            
            srslabel_np = srslabel.cpu().numpy()[0][0]
            srslabel = torch.from_numpy(self.to_categorical(srslabel_np, num_classes=self.n_classes)[np.newaxis, :, :, :, :]).cuda()

            mat_t, code_spa_t = self.spatial_aug.rand_coords(tar_high.shape[2:])
            
            tar_high = self.spatial_aug.augment_spatial(tar_high, mat_t, code_spa_t)
            tar_low = self.spatial_aug.augment_spatial(tar_low, mat_t, code_spa_t)
            tar_high_r = self.spatial_aug.augment_spatial(tar_high_r, mat_t, code_spa_t)
            tar_low_r = self.spatial_aug.augment_spatial(tar_low_r, mat_t, code_spa_t)

            self.train_iterator(srs_high, srs_low, srs_high_r, srs_low_r, srslabel, 
                                tar_high, tar_low, tar_high_r, tar_low_r, epoch, i)

            res = '\t'.join(['Epoch: [%d/%d]' % (epoch + 1, self.epoches),
                             'Iter: [%d/%d]' % (i + 1, self.iters),
                             'Seg: ' + self.L_seg_log.__str__(),
                             'T_Cons: ' + self.L_consist_log.__str__(),
                             'T_Ent: ' + self.L_ent_log.__str__()])
        print(res)
        
        self.model.train()

        return avg_class_dices

    def checkpoint(self, epoch):
        torch.save(self.model.state_dict(), '{0}/ec_epoch_{1}.pth'.format(self.checkpoint_, epoch + self.start_epoch))

    def load_model(self, path, epoch):
        print("loading model epoch ", str(epoch))
        self.model.load_state_dict(torch.load('{0}/ec_epoch_{1}.pth'.format(path, epoch)), strict=True) 

    def train(self):
    
        csv_head = ['epoch', 'loss_consist', 'loss_seg']
        for i in range(self.n_classes - 1):
            csv_head.append(f'val_dice_cls_{i+1}')

        self.trainwriter = LogWriter(name=self.checkpoint_ + "/train_" + self.model_name, head=csv_head) 

        for epoch in range(self.epoches - self.start_epoch):
            self.L_seg_log.reset()
            self.L_consist_log.reset()

            self.epoch = epoch
            self.train_epoch(epoch + self.start_epoch)
            
            self.history['epoch'].append(epoch + self.start_epoch)
            self.history['train_seg_loss'].append(self.L_seg_log.avg) 

            log_list = [
                epoch + self.start_epoch, 
                self.L_consist_log.avg, 
                self.L_seg_log.avg
            ]
            self.trainwriter.writeLog(log_list)

            if epoch % self.save_epoch == 0:
                self.checkpoint(epoch)

        self.checkpoint(self.epoches - self.start_epoch)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='UDA seg Training Function')

    parser.add_argument('--direction', default="A2B")
    parser.add_argument('--part', default="40_20")
    parser.add_argument('--srs_rmmax', type=int, default=40)
    parser.add_argument('--tar_rmmax', type=int, default=20)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--num_epoch', type=int, default=300)
    parser.add_argument('--num_iters', type=int, default=150)
    parser.add_argument('--save_epoch', type=int, default=50)
    parser.add_argument('--model_name', default="CDFreqNet")

    parser.add_argument('--lr_seg', type=int, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--A_root', default="./data/private/datasets/CT/abd/dataTr/")
    parser.add_argument('--B_root', default="./data/private/datasets/MR/abd/dataTr/")

    parser.add_argument('--checkpoint_root', default="/public/home/zhangzengmin/CDFreqNet/checks/checks_abd_ct2mr") 
    
    args = parser.parse_args()

    trainer = Trainer(args = args)
    trainer.train()
