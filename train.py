import os
import random
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
from utils.DynamicLossConstraint import DynamicLossConstraint, SpatialWeighted_DiceLoss


def seed_everything(seed):
    """Seed Python, NumPy, PyTorch, CUDA, and cuDNN before training starts."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    """Seed Python and NumPy inside each DataLoader worker."""
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed):
    """Create an independent deterministic generator for one DataLoader."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


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

        data_listA = [os.path.join(A_root, k) for k in sorted(os.listdir(A_root)) if not k.startswith('.')]
        data_listB = [os.path.join(B_root, k) for k in sorted(os.listdir(B_root)) if not k.startswith('.')]
        data_list_val = [os.path.join(Val_root, k) for k in sorted(os.listdir(Val_root)) if not k.startswith('.')]

        train_srs = data_listA          # Source training set with labels
        train_tar = data_listB          # Target training set without labels
        val_data = data_list_val        # Source validation set with labels

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")

        self.checkpoint_dir = os.path.join(
            args.checkpoint_root,
            self.model_name + '_' + timestamp + '_' + str(self.direction) + '_' + str(self.part)
        )
        crt_file(self.checkpoint_dir)

        self.checkpoint_ = self.checkpoint_dir + "/fold_" + str(args.fold)
        crt_file(self.checkpoint_)

        # Only source-domain validation Dice is used for model selection
        self.best_val_dice = -1.0
        self.best_epoch = 0

        # Data augmentation
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

        # Initialize model
        self.model = CDFreqNet(input_channels=1, num_classes=self.n_classes).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_seg, weight_decay=1e-4)
        self.stn = SpatialTransformer()

        # Training datasets
        trainsrs_dataset = TrainDataset(train_srs, rmmax=self.srs_rmmax, max_move=0.3)
        traintar_dataset = TrainDataset(train_tar, rmmax=self.tar_rmmax, max_move=0.3)

        source_generator = make_generator(args.seed)
        target_generator = make_generator(args.seed + 1)
        validation_generator = make_generator(args.seed + 2)

        self.dataloader_srstrain = DataLoader(
            trainsrs_dataset,
            batch_size=self.bs,
            shuffle=True,
            drop_last=True,
            num_workers=1,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=source_generator
        )

        self.dataloader_tartrain = DataLoader(
            traintar_dataset,
            batch_size=self.bs,
            shuffle=True,
            drop_last=True,
            num_workers=1,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=target_generator
        )

        # Source validation dataset
        val_dataset = Dataset3D(val_data)
        self.dataloader_val = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=validation_generator
        )

        # Define loss
        self.L_seg = dice_loss
        self.L_mse = nn.MSELoss()

        self.dlc = DynamicLossConstraint(num_classes=self.n_classes,
                                             tau=0.5,
                                             feat_channels=64).cuda()

        self.criterion_seg = SpatialWeighted_DiceLoss(num_classes=self.n_classes).cuda()
        self.criterion_cons = SpatialWeighted_DiceLoss(num_classes=self.n_classes).cuda()

        # Define training logs
        self.L_seg_log = AverageMeter(name='L_Seg')
        self.L_consist_log = AverageMeter(name='L_consist')
        self.L_ent_log = AverageMeter(name='L_ent')

        # Define source validation logs
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

    def train_iterator(self,
                       srs_struct,
                       srs_style,
                       srs_struct_r,
                       srs_style_r,
                       srs_label,
                       tar_struct,
                       tar_style,
                       tar_struct_r,
                       tar_style_r,
                       epoch,
                       iters):

        self.optimizer.zero_grad()

        # ============================================================
        # Part 1: Source Domain Training
        # ============================================================
        pred_src_clean, feat_src_clean = self.model(
            x_struct=srs_struct,
            x_style=srs_style,
            mod='A',
            rmmax=self.srs_rmmax
        )

        pred_src_style, feat_src_style = self.model(
            x_struct=srs_struct_r,
            x_style=srs_style_r,
            mod='A',
            rmmax=self.srs_rmmax
        )

        # ============================================================
        # Part 2: Target Domain Unlabeled Training
        # ============================================================
        pred_tar_clean, feat_tar_clean = self.model(
            x_struct=tar_struct,
            x_style=tar_style,
            mod='B',
            rmmax=self.tar_rmmax
        )

        pred_tar_style, feat_tar_aug = self.model(
            x_struct=tar_struct_r,
            x_style=tar_style_r,
            mod='B',
            rmmax=self.tar_rmmax
        )

        w_src, w_tgt, loss_align = self.dlc(
            f_clean_src=feat_src_clean,
            f_aug_src=feat_src_style,
            p_aug_src=pred_src_style,
            mask_src=srs_label,
            f_clean_tgt=feat_tar_clean,
            f_aug_tgt=feat_tar_aug,
            p_aug_tgt=pred_tar_style,
            mask_tgt=pred_tar_clean.detach(),
            current_epoch=epoch
        )

        # Source supervised losses
        loss_seg_src_clean = self.criterion_seg(
            pred_src_clean,
            srs_label,
            weight_map=None
        )

        loss_seg_src = self.criterion_seg(
            pred_src_style,
            srs_label,
            weight_map=w_src
        )

        # Target consistency loss
        loss_cons_tar = self.criterion_seg(
            pred_tar_clean,
            pred_tar_style,
            weight_map=w_tgt
        ) * 0.5

        # Total loss
        loss_total = loss_seg_src + loss_seg_src_clean + loss_cons_tar
        loss_total.backward()
        self.optimizer.step()

        # Logs
        self.L_seg_log.update(loss_seg_src.item(), srs_struct.size(0))
        self.L_consist_log.update(
            loss_cons_tar.item() if torch.is_tensor(loss_cons_tar) else loss_cons_tar,
            srs_struct.size(0)
        )

    def train_epoch(self, epoch, epoch_number):
        self.model.train()

        loader_src = iter(self.dataloader_srstrain)
        loader_tar = iter(self.dataloader_tartrain)

        for i in range(self.iters):

            # Source training data
            try:
                srs_struct, srs_style, srs_struct_r, srs_style_r, srslabel = next(loader_src)
            except StopIteration:
                loader_src = iter(self.dataloader_srstrain)
                srs_struct, srs_style, srs_struct_r, srs_style_r, srslabel = next(loader_src)

            # Target unlabeled training data
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

            # Source augmentation
            mat, code_spa = self.spatial_aug.rand_coords(srs_struct.shape[2:])

            srs_struct = self.spatial_aug.augment_spatial(srs_struct, mat, code_spa)
            srs_style = self.spatial_aug.augment_spatial(srs_style, mat, code_spa)
            srs_struct_r = self.spatial_aug.augment_spatial(srs_struct_r, mat, code_spa)
            srs_style_r = self.spatial_aug.augment_spatial(srs_style_r, mat, code_spa)

            srslabel = self.spatial_aug.augment_spatial(
                srslabel,
                mat,
                code_spa,
                mode="nearest"
            ).int()

            # Label processing
            srslabel_np = srslabel.cpu().numpy()[0][0]
            srslabel = torch.from_numpy(
                self.to_categorical(
                    srslabel_np,
                    num_classes=self.n_classes
                )[np.newaxis, :, :, :, :]
            ).cuda()

            # Target augmentation
            mat_t, code_spa_t = self.spatial_aug.rand_coords(tar_struct.shape[2:])

            tar_struct = self.spatial_aug.augment_spatial(tar_struct, mat_t, code_spa_t)
            tar_style = self.spatial_aug.augment_spatial(tar_style, mat_t, code_spa_t)
            tar_struct_r = self.spatial_aug.augment_spatial(tar_struct_r, mat_t, code_spa_t)
            tar_style_r = self.spatial_aug.augment_spatial(tar_style_r, mat_t, code_spa_t)

            # Train step
            self.train_iterator(
                srs_struct,
                srs_style,
                srs_struct_r,
                srs_style_r,
                srslabel,
                tar_struct,
                tar_style,
                tar_struct_r,
                tar_style_r,
                epoch,
                i
            )

            res = '\t'.join([
                'Epoch: [%d/%d]' % (epoch_number, self.epoches),
                'Iter: [%d/%d]' % (i + 1, self.iters),
                'Seg: ' + self.L_seg_log.__str__(),
                'T_Cons: ' + self.L_consist_log.__str__()
            ])

        print(res)

    def validate(self, epoch_number):
        """
        Validation is performed only on the labeled source-domain validation set.
        The source validation Dice is used for checkpoint selection.
        """
        self.model.eval()
        self.L_val_dice_log.reset()
        self.L_val_loss_log.reset()

        val_class_dices = []

        with torch.no_grad():
            for i, (val_struct, val_style, vallabel) in enumerate(self.dataloader_val):

                if torch.cuda.is_available():
                    val_struct = val_struct.cuda()
                    val_style = val_style.cuda()
                    vallabel = vallabel.cuda()

                vallabel_np_for_loss = vallabel.cpu().numpy()[0][0]

                vallabel_onehot = torch.from_numpy(
                    self.to_categorical(
                        vallabel_np_for_loss,
                        num_classes=self.n_classes
                    )[np.newaxis, :, :, :, :]
                ).cuda()

                pred_mask, _ = self.model(
                    x_struct=val_struct,
                    x_style=val_style,
                    mod='A',
                    rmmax=self.srs_rmmax
                )

                loss_seg = self.L_seg(pred_mask, vallabel_onehot)
                self.L_val_loss_log.update(loss_seg.item(), val_struct.size(0))

                valseg_np = np.argmax(pred_mask[0].cpu().numpy(), axis=0)
                valseg_onehot = self.to_categorical(
                    valseg_np,
                    num_classes=self.n_classes
                )
                vallab_onehot_np = vallabel_onehot[0].cpu().numpy()

                valdices_all = []

                for cls in range(self.n_classes - 1):
                    valdices_all.append(
                        dice(
                            valseg_onehot[cls + 1],
                            vallab_onehot_np[cls + 1]
                        )
                    )

                val_class_dices.append(valdices_all)

                mean_dice = np.mean(valdices_all)
                self.L_val_dice_log.update(mean_dice, val_struct.size(0))

        if len(val_class_dices) > 0:
            avg_class_dices = np.mean(np.array(val_class_dices), axis=0)
            class_dice_str = " | ".join([
                f"Cls{i + 1}: {d:.4f}"
                for i, d in enumerate(avg_class_dices)
            ])
        else:
            avg_class_dices = np.zeros(self.n_classes - 1)
            class_dice_str = "No Data"

        print(
            f"Source Validation Epoch {epoch_number}: "
            f"Val_Loss {self.L_val_loss_log.avg:.4f}, "
            f"Val_Dice {self.L_val_dice_log.avg:.4f}"
        )
        print(f"Details Dice: {class_dice_str}")
        print("")

        self.model.train()

        return avg_class_dices

    def draw_curve(self):
        epochs = self.history['epoch']
        seg_loss = self.history['train_seg_loss']
        val_dice = self.history['val_dice']

        fig, ax1 = plt.subplots(figsize=(10, 6))

        color_loss = 'tab:blue'
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Segmentation Loss', color=color_loss, fontsize=12)
        l1, = ax1.plot(
            epochs,
            seg_loss,
            color=color_loss,
            label='Train Seg Loss',
            linewidth=2
        )
        ax1.tick_params(axis='y', labelcolor=color_loss)
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()

        color_dice = 'tab:red'
        ax2.set_ylabel('Source Validation Dice', color=color_dice, fontsize=12)
        l2, = ax2.plot(
            epochs,
            val_dice,
            color=color_dice,
            label='Source Val Dice',
            linewidth=2
        )
        ax2.tick_params(axis='y', labelcolor=color_dice)

        plt.title('Training Loss & Source Validation Dice', fontsize=14)

        lines = [l1, l2]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='center right')

        fig.tight_layout()

        save_path = os.path.join(self.checkpoint_, 'training_curves.png')
        plt.savefig(save_path, dpi=100)
        plt.close()

    def checkpoint(self, epoch_number):
        save_path = os.path.join(
            self.checkpoint_,
            'best_source_val_model.pth'
        )

        torch.save(
            self.model.state_dict(),
            save_path
        )

        with open(
            os.path.join(self.checkpoint_, 'best_source_val.txt'),
            'w'
        ) as f:
            f.write('Best epoch: %d\n' % epoch_number)
            f.write('Best source validation Dice: %.6f\n' % self.best_val_dice)

        print(
            'Best model saved: epoch %d, source validation Dice %.4f'
            % (epoch_number, self.best_val_dice)
        )

    def load_model(self, path):
        print("loading model: ", path)
        self.model.load_state_dict(
            torch.load(path),
            strict=True
        )

    def train(self):

        csv_head = [
            'epoch',
            'loss_consist',
            'loss_seg',
            'source_val_dice',
            'source_val_loss'
        ]

        for i in range(self.n_classes - 1):
            csv_head.append(f'source_val_dice_cls_{i + 1}')

        self.trainwriter = LogWriter(
            name=self.checkpoint_ + "/train_" + self.model_name,
            head=csv_head
        )

        for epoch_index in range(self.start_epoch, self.epoches):

            epoch_number = epoch_index + 1

            self.L_seg_log.reset()
            self.L_consist_log.reset()

            self.epoch = epoch_number

            # Training
            self.train_epoch(epoch_index, epoch_number)

            # Source-domain validation
            per_class_dices = self.validate(epoch_number)

            self.history['epoch'].append(epoch_number)
            self.history['train_seg_loss'].append(self.L_seg_log.avg)
            self.history['val_dice'].append(self.L_val_dice_log.avg)

            self.draw_curve()

            log_list = [
                epoch_number,
                self.L_consist_log.avg,
                self.L_seg_log.avg,
                self.L_val_dice_log.avg,
                self.L_val_loss_log.avg
            ]

            log_list.extend(per_class_dices)

            self.trainwriter.writeLog(log_list)

            # Save only the best checkpoint according to source validation Dice
            if self.L_val_dice_log.avg > self.best_val_dice:
                self.best_val_dice = self.L_val_dice_log.avg
                self.best_epoch = epoch_number
                self.checkpoint(epoch_number)

        print("")
        print("Training finished.")
        print(
            "Best source validation Dice: %.4f at epoch %d"
            % (self.best_val_dice, self.best_epoch)
        )


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
        description='CDFreqNet UDA Training Function'
    )

    parser.add_argument('--fold_num', type=int, default=5)
    parser.add_argument('--fold', type=int, default=0)

    parser.add_argument('--direction', default="A2B")
    parser.add_argument('--part', default="50_30")

    parser.add_argument('--srs_rmmax', type=int, default=50)
    parser.add_argument('--tar_rmmax', type=int, default=30)

    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--num_epoch', type=int, default=300)
    parser.add_argument('--num_iters', type=int, default=100)

    parser.add_argument('--model_name', default="CDFreqNet")
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--lr_seg', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=5)

    # Generic example paths
    parser.add_argument(
        '--A_root',
        default="./data/source/train/"
    )
    parser.add_argument(
        '--B_root',
        default="./data/target/train/"
    )
    parser.add_argument(
        '--Val_root',
        default="./data/source/val/"
    )

    parser.add_argument(
        '--checkpoint_root',
        default="./checkpoints/"
    )

    args = parser.parse_args()

    seed_everything(args.seed)

    trainer = Trainer(args=args)
    trainer.train()
