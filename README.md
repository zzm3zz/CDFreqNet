# CDFreqNet

Official PyTorch implementation of:

**Causally Inspired Decoupled Frequency Intervention for Unsupervised Domain Adaptation in Medical Image Segmentation**

CDFreqNet is a 3D unsupervised domain adaptation framework for cross-domain medical image segmentation. The framework contains three main components:

- **DFI**: Decoupled Frequency Intervention
  - **AFI**: appearance-focused intervention on the low-frequency representation using Density-Guided Remap (DGR)
  - **SFI**: structure-focused intervention on the high-frequency representation using bilateral filtering and stochastic cubic Bézier remapping
- **AFR**: Adaptive Frequency Reassembly
- **DLC**: Dynamic Loss Constraint

The implementation follows the experimental protocol described in the paper: labeled source-domain data are used for supervised learning, unlabeled target-domain data are used for adaptation, and the best checkpoint is selected exclusively on the labeled source-domain validation set.

---

## Repository Structure

```text
CDFreqNet/
├── data/
│   ├── CT/
│   └── MRI/
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
```

The current `data/` directory is organized first by imaging modality. Dataset-specific folders and train/validation/test splits can then be placed under the corresponding modality directory.

For example:

```text
data/
├── CT/
│   ├── BTCV/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── MMWHS/
│       ├── train/
│       ├── val/
│       └── test/
└── MRI/
    ├── CHAOS/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── MMWHS/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── PROMISE12/
    └── BraTS18/
```

The exact folder names can be changed through the command-line arguments in the training and testing scripts.

For the additional PET experiments, PET data can be organized separately, e.g.:

```text
data/
└── PET/
    └── FLARE25/
        ├── train/
        └── test/
```

---

## Environment

The code is implemented in Python and PyTorch.

Main dependencies include:

```text
torch
numpy
scipy
SimpleITK
opencv-python
dtcwt
matplotlib
Pillow
```

Install the appropriate PyTorch version according to your CUDA environment and install the remaining dependencies with `pip`.

---

## Datasets

The datasets used in the main experiments and additional analyses can be obtained from their official websites.

### Abdominal CT

**BTCV (Beyond the Cranial Vault)**  
Official website: https://www.synapse.org/#!Synapse:syn3193805

### Abdominal MRI

**CHAOS Challenge**  
Official website: https://chaos.grand-challenge.org/

### Cardiac CT/MRI

**MM-WHS 2017 (Multi-Modality Whole Heart Segmentation Challenge)**  
Official website: https://zmiclab.github.io/zxh/0/mmwhs/

### Prostate MRI

**PROMISE12 Challenge**  
Official website: https://promise12.grand-challenge.org/

### Pathological Brain MRI

**BraTS 2018 (Multimodal Brain Tumor Segmentation Challenge 2018)**  
Official website: https://www.med.upenn.edu/cbica/brats2018/data.html

BraTS18 is used for the additional pathological cross-sequence adaptation experiments.

### Abdominal CT/PET

**FLARE 2025 Challenge**  
Official website: https://openreview.net/group?id=MICCAI.org/2025/Challenge/FLARE

FLARE25 is used for the additional PET-to-CT and CT-to-PET abdominal cross-modality experiments.

Please follow the license, access requirements, and citation policy specified by each dataset provider. Dataset files are not redistributed in this repository.

---

## Main Preprocessing

The preprocessing implementation for the main CT/MRI experiments is provided in:

```text
preprocess/preprocess_cdfreqnet.py
```

The preprocessing pipeline follows the protocol described in the paper:

```text
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
```

The DTCWT settings are:

```text
N = 2
alpha = 2.0
```

Example for CT:

```bash
python preprocess/preprocess_cdfreqnet.py \
  --image_root ./raw_data/CT/images/ \
  --label_root ./raw_data/CT/labels/ \
  --save_root ./data/CT/BTCV/train/ \
  --modality CT \
  --use_roi \
  --overwrite
```

Example for MRI:

```bash
python preprocess/preprocess_cdfreqnet.py \
  --image_root ./raw_data/MRI/images/ \
  --label_root ./raw_data/MRI/labels/ \
  --save_root ./data/MRI/CHAOS/train/ \
  --modality MRI \
  --use_roi \
  --overwrite
```

For PROMISE12, ROI cropping is not used.

After preprocessing, each NPZ file contains:

```text
data    normalized input volume
high    HF-enhanced representation
low     LF representation
seg     segmentation label
```

The dataloader directly reads the preprocessed `high`, `low`, and `seg` arrays and does not perform a second intensity normalization.

---

## PET/CT Preprocessing

The PET/CT experiments use a different preprocessing protocol from the main abdominal CT/MRI experiments.

For the FLARE25 PET-to-CT and CT-to-PET experiments, annotation-derived foreground bounding boxes are not used for localization. Instead, the volumes are processed using:

```text
Raw CT/PET volume
    ↓
Reorientation
    ↓
Fixed central transverse field of view
    ↓
Spatial normalization
    ↓
DTCWT decomposition
    ↓
high / low representations
```

This fixed spatial preprocessing is independent of target-domain annotations and is used consistently for both adaptation directions.

Because PET intensity values have different physical and statistical characteristics from CT Hounsfield units and conventional MRI intensities, PET data should not be processed using the CT HU clipping window `[-125, 275]`. The PET preprocessing used for the FLARE25 experiments therefore follows the PET-specific spatial/intensity normalization used for this experiment before DTCWT decomposition.

The DTCWT configuration and subsequent CDFreqNet intervention settings remain unchanged:

```text
N = 2
alpha = 2.0
Ms = 50
Mt = 30
eta = 0.3
BF diameter = 7
BF sigma_int = 0.2
BF sigma_sp = 2
```

For PET/CT adaptation, source annotations are used for source-domain training, while target-domain annotations are reserved exclusively for final evaluation.

---

## Data Split and UDA Protocol

For each adaptation direction, the data are divided according to their role rather than by changing the modality-level directory organization.

```text
Source domain:
    labeled training set
    labeled validation set

Target domain:
    unlabeled training set
    held-out test set
```

The released code follows the standard UDA protocol:

- Source training images and labels are used for supervised segmentation learning.
- Target training images participate in UDA training without using their labels.
- Source validation Dice is used for checkpoint selection.
- Target-domain labels are not used for model selection or hyperparameter selection.
- The target test set is used only for final evaluation.

---

## Training

CDFreqNet uses labeled source-domain training data together with unlabeled target-domain training data.

Example:

```bash
python train_abd_ct2mr.py \
  --A_root ./data/CT/BTCV/train/ \
  --B_root ./data/MRI/CHAOS/train/ \
  --Val_root ./data/CT/BTCV/val/ \
  --checkpoint_root ./checkpoints/
```

For the reverse direction, the source and target paths can be exchanged accordingly.

The main settings are:

```text
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
```

The best checkpoint is selected exclusively according to the mean foreground Dice on the source-domain validation set.

The training script saves:

```text
best_source_val_model.pth
```

No target-domain test result is used during training or checkpoint selection.

---

## Testing

The target-domain test set is evaluated only after training and source-domain checkpoint selection have been completed.

Example:

```bash
python test_abd_ct2mr.py \
  --weight_path ./checkpoints/best_source_val_model.pth \
  --test_dir ./data/MRI/CHAOS/test/
```

The testing script reports segmentation metrics and saves the predicted masks.

---

## Decoupled Frequency Intervention

### Appearance-Focused Intervention

Density-Guided Remap (DGR) is applied to the low-frequency representation.

```text
Ms = 50
Mt = 30
eta = 0.3
```

For each randomly generated intensity partition, DGR computes the normalized voxel occupancy and restricts the admissible displacement of densely occupied intervals while allowing larger perturbations for sparse intervals.

### Structure-Focused Intervention

The high-frequency representation is processed using slice-wise bilateral filtering followed by stochastic cubic Bézier remapping.

```text
Bilateral-filter diameter     7
Intensity scale               0.2
Spatial scale                 2
```

### Dynamic Loss Constraint

DLC uses voxel-wise prediction confidence and intervention-induced feature deviation with epoch-wise annealing:

```text
W_DLC = C^omega(t) * (1 + omega(t) * D_cos)
```

where `omega(t)` gradually increases from 0 to 1 during training.

---

## Additional Experiments

In addition to the main abdominal CT/MRI, cardiac CT/MRI, and prostate cross-site experiments, the paper includes:

- BraTS18 FLAIR-to-T2 and T2-to-FLAIR pathological cross-sequence adaptation.
- FLARE25 CT-to-PET and PET-to-CT abdominal cross-modality adaptation.
- Multi-source cross-site prostate adaptation.
- Learnable wavelet decomposition analysis.
- Annotation-free coarse-to-fine preprocessing experiments.
- Hyperparameter and implementation analyses.

These results are reported in the public **Extended Experimental Material** available from the repository Releases page.

---

## Notes

- Dataset files are not distributed with this repository.
- Please download all datasets from their official websites.
- Dataset-specific licenses and usage agreements remain the responsibility of the user.
- The main `preprocess/preprocess_cdfreqnet.py` script corresponds to the CT/MRI preprocessing pipeline used in the primary experiments.
- PET preprocessing is handled separately because PET intensity statistics and spatial characteristics differ from conventional CT/MRI data.
- The released preprocessing, training, validation, and testing code is intended to reproduce the experimental protocol described in the paper.
