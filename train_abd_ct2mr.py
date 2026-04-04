import os
from datetime import datetime
import torch
from torch import nn
from models.network import CDFreqNet
from utils.STN import SpatialTransformer
from utils.Transform_self import SpatialTransform
from utils.dataloader import Dataset3D_DFI as TrainDataset
from utils.dataloader import Dataset3D  # Import for validation dataset
from torch.utils.data import DataLoader
from utils.losses import dice_loss, prob_entropyloss
from utils.utils import AverageMeter, LogWriter, dice
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
# [New] 导入 DDSA
from utils.DynamicTemporalConstraint import DynamicTemporalConstraint, SpatialWeighted_DiceLoss

def crt_file(path):
    os.makedirs(path, exist_ok=True)

class Trainer(object):
    def __init__(self, args=None):
        super(Trainer, self).__init__()

        self.fold_num = args.fold_num
        self.fold = args.fold
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
        self.checkpoint_ = self.checkpoint_dir + "/fold_" + str(args.fold)
        crt_file(self.checkpoint_)

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
        # 验证集通常不需要多进程，除非验证非常慢
        self.dataloader_val = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

        # define loss
        self.L_seg = dice_loss
        self.L_mse = nn.MSELoss() 
        
        self.dtc = DynamicTemporalConstraint(num_classes=self.n_classes, 
                                      tau=0.5, feat_channels=64).cuda()
        # [New] 加权 Loss
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

    def train_iterator(self, srs_struct, srs_style, srs_struct_r, srs_style_r, srs_label, 
                       tar_struct, tar_style, tar_struct_r, tar_style_r, epoch, iters):
        """
        核心训练循环：显式接收解耦后的 Struct/Style 数据
        """
        self.optimizer.zero_grad()

        # ============================================================
        # Part 1: Source Domain Training (Supervised + Consistency)
        # ============================================================
        pred_src_clean, feat_src_clean = self.model(x_struct=srs_struct, x_style=srs_style, 
                                    mod='A', rmmax=self.srs_rmmax)
        pred_src_style, feat_src_style = self.model(x_struct=srs_struct_r, x_style=srs_style_r, 
                                    mod='A', rmmax=self.srs_rmmax)

        pred_tar_clean, feat_tar_clean = self.model(x_struct=tar_struct, x_style=tar_style, 
                                    mod='B', rmmax=self.tar_rmmax)
        pred_tar_style, feat_tar_aug = self.model(x_struct=tar_struct_r, x_style=tar_style_r, 
                                    mod='B', rmmax=self.tar_rmmax)
        
        w_src, w_tgt, loss_align = self.dtc(
            f_clean_src=feat_src_clean, 
            f_aug_src=feat_src_style, 
            p_aug_src=pred_src_style,     # <--- 新增: 传入 Source Aug 的预测
            mask_src=srs_label,
            f_clean_tgt=feat_tar_clean, 
            f_aug_tgt=feat_tar_aug, 
            p_aug_tgt=pred_tar_style, 
            mask_tgt=pred_tar_clean.detach(),
            current_epoch=epoch
        )
        # 1.3 Source Losses
        # loss_seg_src_clean = self.L_seg(pred_src_clean, srs_label)
        loss_seg_src_clean = self.criterion_seg(pred_src_clean, srs_label, weight_map=None)
                           
        # loss_seg_src = self.L_seg(pred_src_style, srs_label) 
        loss_seg_src = self.criterion_seg(pred_src_style, srs_label, weight_map=w_src)
                           
        # loss_cons_src = self.L_seg(pred_src_style, pred_src_clean) * 0.1

        # 2.3 Target Consistency Loss
        # loss_cons_tar = self.L_seg(pred_tar_style, pred_tar_clean) * 0.5
        loss_cons_tar = self.criterion_seg(pred_tar_clean, pred_tar_style, weight_map=w_tgt) * 0.1

        # ============================================================
        # Total Loss & Backward
        # ============================================================
        w_src_seg = 1.0
        w_cons = 1.0

        loss_total = w_src_seg * (loss_seg_src + loss_seg_src_clean) + \
                     w_cons * (loss_cons_tar)
        loss_total.backward()
        self.optimizer.step()

        # Logs
        self.L_seg_log.update(loss_seg_src.item(), srs_struct.size(0))
        self.L_consist_log.update(loss_cons_tar.item() if torch.is_tensor(loss_cons_tar) else loss_cons_tar, srs_struct.size(0))

    def train_epoch(self, epoch):
        self.model.train()
        
        # 创建迭代器
        loader_src = iter(self.dataloader_srstrain)
        loader_tar = iter(self.dataloader_tartrain)
        
        # 迭代次数以 Source 为准
        for i in range(self.iters):
            # 获取 Source 数据 (5个项目)
            try:
                # [关键] 这里需要对应 dataloader 返回的顺序
                # img, img_r, struct, style, label
                srs_struct, srs_style, srs_struct_r, srs_style_r, srslabel = next(loader_src)
            except StopIteration:
                loader_src = iter(self.dataloader_srstrain)
                srs_struct, srs_style, srs_struct_r, srs_style_r, srslabel = next(loader_src)
                
            # 获取 Target 数据 (5个项目, 忽略 label 和 img_r)
            try:
                tar_struct, tar_style, tar_struct_r, tar_style_r, _ = next(loader_tar)
            except StopIteration:
                loader_tar = iter(self.dataloader_tartrain)
                tar_struct, tar_style, tar_struct_r, tar_style_r, _ = next(loader_tar)

            # Move to GPU
            if torch.cuda.is_available():
                srs_struct = srs_struct.cuda()
                srs_style = srs_style.cuda()
                srs_struct_r = srs_struct_r.cuda()
                srs_style_r = srs_style_r.cuda()
                srslabel = srslabel.cuda()
                
                tar_struct = tar_struct.cuda()
                tar_style = tar_style.cuda()
                tar_struct_r = tar_struct_r.cuda()
                tar_style_r = tar_style_r.cuda()
                
            # --- Source Augmentation (同步空间变换) ---
            # 必须使用同一个 mat, code 对原图及其分解图做变换，保证对齐
           
            mat, code_spa = self.spatial_aug.rand_coords(srs_struct.shape[2:])
            
            srs_struct = self.spatial_aug.augment_spatial(srs_struct, mat, code_spa)
            srs_style = self.spatial_aug.augment_spatial(srs_style, mat, code_spa)
            srs_struct_r = self.spatial_aug.augment_spatial(srs_struct_r, mat, code_spa)
            srs_style_r = self.spatial_aug.augment_spatial(srs_style_r, mat, code_spa)
            
            srslabel = self.spatial_aug.augment_spatial(srslabel, mat, code_spa, mode="nearest").int()
            
            # Label processing
            srslabel_np = srslabel.cpu().numpy()[0][0]
            srslabel = torch.from_numpy(self.to_categorical(srslabel_np, num_classes=self.n_classes)[np.newaxis, :, :, :, :]).cuda()

            # --- Target Augmentation (同步空间变换) ---
            mat_t, code_spa_t = self.spatial_aug.rand_coords(tar_struct.shape[2:])
            
            tar_struct = self.spatial_aug.augment_spatial(tar_struct, mat_t, code_spa_t)
            tar_style = self.spatial_aug.augment_spatial(tar_style, mat_t, code_spa_t)
            tar_struct_r = self.spatial_aug.augment_spatial(tar_struct_r, mat_t, code_spa_t)
            tar_style_r = self.spatial_aug.augment_spatial(tar_style_r, mat_t, code_spa_t)
            
            # Train Step (传入新的分解数据)
            self.train_iterator(srs_struct, srs_style, srs_struct_r, srs_style_r, srslabel, 
                                tar_struct, tar_style, tar_struct_r, tar_style_r, epoch, i)

            # Print Log
            res = '\t'.join(['Epoch: [%d/%d]' % (epoch + 1, self.epoches),
                             'Iter: [%d/%d]' % (i + 1, self.iters),
                             'Seg: ' + self.L_seg_log.__str__(),
                             'T_Cons: ' + self.L_consist_log.__str__(),
                             'T_Ent: ' + self.L_ent_log.__str__()])
        print(res)
    
    def validate(self, epoch):
        self.model.eval()
        self.L_val_dice_log.reset()
        self.L_val_loss_log.reset()

        val_class_dices = [] 

        with torch.no_grad():
            for i, (tar_struct, tar_style, tarlabel) in enumerate(self.dataloader_val):
                if torch.cuda.is_available():
                    tar_struct = tar_struct.cuda()
                    tar_style = tar_style.cuda()
                    tarlabel = tarlabel.cuda()

                tarlabel_np_for_loss = tarlabel.cpu().numpy()[0][0] 
                tarlabel_onehot = torch.from_numpy(
                    self.to_categorical(tarlabel_np_for_loss, num_classes=self.n_classes)[np.newaxis, :, :, :, :]
                ).cuda() 

                pred_mask_b, _ = self.model(x_struct=tar_struct, x_style=tar_style) 
                
                loss_seg = self.L_seg(pred_mask_b, tarlabel_onehot)
                self.L_val_loss_log.update(loss_seg.item(), tar_struct.size(0))

                tarseg_np = np.argmax(pred_mask_b[0].cpu().numpy(), axis=0) 
                tarseg_onehot = self.to_categorical(tarseg_np, num_classes=self.n_classes) 
                tarlab_onehot_np = tarlabel_onehot[0].cpu().numpy() 

                tardice_all = []
                # 计算除背景外的每一类的 Dice
                for cls in range(self.n_classes - 1):
                    tardice_all.append(dice(tarseg_onehot[cls + 1], tarlab_onehot_np[cls + 1]))

                val_class_dices.append(tardice_all)

                mean_dice = np.mean(tardice_all)
                self.L_val_dice_log.update(mean_dice, tar_struct.size(0))

        if len(val_class_dices) > 0:
            avg_class_dices = np.mean(np.array(val_class_dices), axis=0)
            class_dice_str = " | ".join([f"Cls{i+1}: {d:.4f}" for i, d in enumerate(avg_class_dices)])
        else:
            avg_class_dices = np.zeros(self.n_classes - 1)
            class_dice_str = "No Data"

        print(f"Validation Epoch {epoch}: Val_Loss {self.L_val_loss_log.avg:.4f}, Val_Dice {self.L_val_dice_log.avg:.4f}")
        print(f"Details Dice: {class_dice_str}")
        print("")
        self.model.train()
        
        # [修改] 返回每个类别的平均 Dice
        return avg_class_dices

    def draw_curve(self):
        epochs = self.history['epoch']
        seg_loss = self.history['train_seg_loss']
        val_dice = self.history['val_dice']

        fig, ax1 = plt.subplots(figsize=(10, 6))
        color_loss = 'tab:blue'
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Segmentation Loss', color=color_loss, fontsize=12)
        l1, = ax1.plot(epochs, seg_loss, color=color_loss, label='Train Seg Loss', linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color_loss)
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()  
        color_dice = 'tab:red'
        ax2.set_ylabel('Validation Dice', color=color_dice, fontsize=12) 
        l2, = ax2.plot(epochs, val_dice, color=color_dice, label='Val Dice', linewidth=2)
        ax2.tick_params(axis='y', labelcolor=color_dice)
        
        plt.title('Training Loss & Validation Dice', fontsize=14)
        lines = [l1, l2]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='center right')
        fig.tight_layout() 
        
        save_path = os.path.join(self.checkpoint_, 'training_curves.png')
        plt.savefig(save_path, dpi=100)
        plt.close()

    def checkpoint(self, epoch):
        torch.save(self.model.state_dict(), '{0}/ec_epoch_{1}.pth'.format(self.checkpoint_, epoch + self.start_epoch))

    def load_model(self, path, epoch):
        print("loading model epoch ", str(epoch))
        self.model.load_state_dict(torch.load('{0}/ec_epoch_{1}.pth'.format(path, epoch)), strict=True) 

    def train(self):
        # [修改] 动态构建 CSV 表头
        # 基础表头
        csv_head = ['epoch', 'loss_consist', 'loss_seg', 'val_dice', 'val_loss']
        # 为每个前景类别添加表头 (Cls_1, Cls_2, ...)
        for i in range(self.n_classes - 1):
            csv_head.append(f'val_dice_cls_{i+1}')

        self.trainwriter = LogWriter(name=self.checkpoint_ + "/train_" + self.model_name, head=csv_head) 

        for epoch in range(self.epoches - self.start_epoch):
            self.L_seg_log.reset()
            self.L_consist_log.reset()

            self.epoch = epoch
            self.train_epoch(epoch + self.start_epoch)
            
            # [修改] 接收 validate 返回的每个类别的 Dice
            per_class_dices = self.validate(epoch + self.start_epoch)
            
            self.history['epoch'].append(epoch + self.start_epoch)
            self.history['train_seg_loss'].append(self.L_seg_log.avg) 
            self.history['val_dice'].append(self.L_val_dice_log.avg) 
            self.draw_curve() 
            
            # [修改] 构建写入日志的数据列表
            log_list = [
                epoch + self.start_epoch, 
                self.L_consist_log.avg, 
                self.L_seg_log.avg, 
                self.L_val_dice_log.avg, 
                self.L_val_loss_log.avg 
            ]
            
            # [修改] 将每个类别的 Dice 追加到日志列表中
            log_list.extend(per_class_dices)

            self.trainwriter.writeLog(log_list)

            if epoch % self.save_epoch == 0:
                self.checkpoint(epoch)

        self.checkpoint(self.epoches - self.start_epoch)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='UDA seg Training Function')

    parser.add_argument('--fold_num', type=int, default=5)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--direction', default="A2B")
    parser.add_argument('--part', default="40_20")
    parser.add_argument('--srs_rmmax', type=int, default=40)
    parser.add_argument('--tar_rmmax', type=int, default=20)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--num_epoch', type=int, default=300)
    parser.add_argument('--num_iters', type=int, default=150)
    parser.add_argument('--save_epoch', type=int, default=1)
    parser.add_argument('--model_name', default="CDFreqNet")

    parser.add_argument('--lr_seg', type=int, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--A_root', default="/public/home/zhangzengmin/DCLPS-main/data/private/datasets/CT/abd/dataTr/")
    parser.add_argument('--B_root', default="/public/home/zhangzengmin/DCLPS-main/data/private/datasets/MR/abd/dataTr/")
    parser.add_argument('--Val_root', default="/public/home/zhangzengmin/DCLPS-main/data/private/datasets/MR/abd/dataTs/")

    parser.add_argument('--checkpoint_root', default="/public/home/zhangzengmin/CDFreqNet/checks/checks_abd_ct2mr_norm") 
    
    args = parser.parse_args()

    trainer = Trainer(args = args)
    trainer.train()