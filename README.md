CDFreqNet

Official PyTorch implementation of:

Causally Inspired Decoupled Frequency Intervention for Unsupervised Domain Adaptation in Medical Image Segmentation

CDFreqNet is a 3D unsupervised domain adaptation framework for cross-domain medical image segmentation. The framework contains three main components:

DFI: Decoupled Frequency Intervention

AFI: appearance-focused intervention on the low-frequency representation using Density-Guided Remap (DGR)

SFI: structure-focused intervention on the high-frequency representation using bilateral filtering and stochastic cubic Bézier remapping

AFR: Adaptive Frequency Reassembly

DLC: Dynamic Loss Constraint

The implementation follows the training and model-selection protocol described in the paper: labeled source-domain data are used for supervised training, unlabeled target-domain data are used for UDA training, and the best checkpoint is selected exclusively on the labeled source-domain validation set.

Repository Structure

CDFreqNet/
├── models/
│   ├── network.py
│   └── ...
├── preprocess/
│   └── preprocess_cdfreqnet.py
├── utils/
│   ├── dataloader.py
│   ├── DynamicLossConstraint.py
│   └── ...
├── train_abd_ct2mr.py
├── test_abd_ct2mr.py
└── README.md

Environment

The code is implemented in Python and PyTorch.

Main dependencies include:

torch
numpy
scipy
SimpleITK
opencv-python
dtcwt
matplotlib
Pillow

Install the required packages according to your local CUDA/PyTorch environment.

Datasets

The datasets used in the main experiments can be obtained from their official websites.

Abdominal CT

BTCV (Beyond the Cranial Vault)
Official website: https://www.synapse.org/#!Synapse:syn3193805

Abdominal MRI

CHAOS Challenge
Official website: https://chaos.grand-challenge.org/

Cardiac CT/MRI

MM-WHS 2017 (Multi-Modality Whole Heart Segmentation Challenge)
Official website: https://zmiclab.github.io/zxh/0/mmwhs/

Prostate MRI

PROMISE12 Challenge
Official website: https://promise12.grand-challenge.org/

Please follow the licenses, access requirements, and citation policies specified by the original dataset providers.

Preprocessing

The preprocessing implementation is provided in:

preprocess/preprocess_cdfreqnet.py

The preprocessing pipeline follows the protocol described in the paper:

Raw 3D volume
    ↓
RAI orientation
    ↓
In-plane resampling to 1.0 × 1.0 mm²
(original z-axis spacing is preserved)
    ↓
Intensity preprocessing
    ├── CT: clip to [-125, 275] HU
    └── MRI: clip at the 99.5th percentile
    ↓
Min-max normalization to [-1, 1]
    ↓
3D ROI cropping with a 15-voxel margin
(abdominal and cardiac datasets)
    ↓
Resize to 64 × 160 × 160
    ↓
Two-level 3D DTCWT
    ↓
Save data / high / low / seg as NPZ

The DTCWT decomposition uses:

N = 2
alpha = 2.0

Example for CT:

python preprocess/preprocess_cdfreqnet.py \
  --image_root ./raw_data/CT/images/ \
  --label_root ./raw_data/CT/labels/ \
  --save_root ./data/CT/preprocessed/ \
  --modality CT \
  --use_roi \
  --overwrite

Example for MRI:

python preprocess/preprocess_cdfreqnet.py \
  --image_root ./raw_data/MRI/images/ \
  --label_root ./raw_data/MRI/labels/ \
  --save_root ./data/MRI/preprocessed/ \
  --modality MRI \
  --use_roi \
  --overwrite

For PROMISE12, ROI cropping is not used.

After preprocessing, each NPZ file contains:

data    normalized input volume
high    HF-enhanced representation
low     LF representation
seg     segmentation label

Data Organization

A recommended organization is:

data/
├── source/
│   ├── train/
│   └── val/
└── target/
    ├── train/
    └── test/

For each adaptation direction:

source/train: labeled source-domain training data

source/val: labeled source-domain validation data

target/train: unlabeled target-domain training data

target/test: target-domain data reserved for final evaluation

Target-domain labels are not used during UDA training or checkpoint selection.

Training

CDFreqNet uses labeled source-domain training data and unlabeled target-domain training data.

The best checkpoint is selected only according to the mean foreground Dice on the source-domain validation set.

Example:

python train_abd_ct2mr.py \
  --A_root ./data/source/train/ \
  --B_root ./data/target/train/ \
  --Val_root ./data/source/val/ \
  --checkpoint_root ./checkpoints/

The main settings are:

Epochs                    300
Batch size                1
Learning rate             1e-3
DTCWT level N             2
HF sharpening alpha       2.0
Source DGR interval Ms    50
Target DGR interval Mt    30
DGR eta                   0.3
BF diameter               7
BF sigma_int              0.2
BF sigma_sp               2
Consistency weight        0.5

The training script saves only the checkpoint with the best source-domain validation Dice:

best_source_val_model.pth

Testing

The target-domain test set is used only after training and checkpoint selection are completed.

Example:

python test_abd_ct2mr.py \
  --weight_path ./checkpoints/best_source_val_model.pth \
  --test_dir ./data/target/test/

The test script reports Dice and ASD and saves the predicted segmentation masks.

UDA Protocol

The released code follows the standard unsupervised domain adaptation setting:

Source domain:
    training images + labels
    validation images + labels

Target domain:
    unlabeled training images
    test images + labels for final evaluation only

Specifically:

Source labels are used for supervised segmentation training.

Target training labels are not accessed by the training dataloader.

Target images participate in UDA training without labels.

Source validation Dice is used for checkpoint selection.

Target test labels are used only for final quantitative evaluation.

Main DFI Settings

Appearance-Focused Intervention

DGR is applied to the low-frequency representation.

Ms = 50
Mt = 30
eta = 0.3

DGR computes the normalized voxel occupancy of each intensity interval and adaptively restricts interval displacement according to its density.

Structure-Focused Intervention

The high-frequency representation is processed using slice-wise bilateral filtering followed by stochastic cubic Bézier remapping.

Bilateral-filter diameter     7
Intensity scale               0.2
Spatial scale                 2

Dynamic Loss Constraint

DLC uses voxel-wise prediction confidence and intervention-induced feature deviation with epoch-wise annealing:

W_DLC = C^omega(t) * (1 + omega(t) * D_cos)

where omega(t) gradually increases from 0 to 1 during training.

Extended Experimental Material

Additional experimental results, hyperparameter analyses, annotation-free preprocessing experiments, learnable-frequency comparisons, pathological adaptation experiments, and generalization analyses are provided in the Extended Experimental Material available from the repository Releases page.

Notes

Dataset files are not redistributed in this repository.

Please download each dataset from its official website and follow the corresponding usage agreement.

The released preprocessing, training, validation, and testing code is intended to reproduce the experimental protocol described in the paper.
