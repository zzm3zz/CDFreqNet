# CDFreqNet

Official PyTorch implementation of:

**Causally Inspired Decoupled Frequency Intervention for Unsupervised Domain Adaptation in Medical Image Segmentation**

The implementation follows the experimental protocol described in the paper: labeled source-domain data are used for supervised learning, unlabeled target-domain images are used for adaptation, and the best checkpoint is selected exclusively on the labeled source-domain validation set. No target-domain labels are used for optimization, hyperparameter selection, or checkpoint selection.

---

## Repository Structure

```text
CDFreqNet/
├── data/
│   ├── CT/
│   ├── MRI/
│   └── PET/
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

The datasets used in our experiments can be obtained from their official websites.

### Abdominal CT

**BTCV**  
Official: https://www.synapse.org/#!Synapse:syn3193805

### Abdominal MRI

**CHAOS**  
Official: https://chaos.grand-challenge.org/

### Cardiac CT/MRI

**MM-WHS 2017**  
Official: https://zmiclab.github.io/zxh/0/mmwhs/

### Prostate MRI

**PROMISE12**  
Official: https://promise12.grand-challenge.org/

### Pathological Brain MRI

**BraTS 2018**  
Official: https://www.med.upenn.edu/cbica/brats2018/data.html

BraTS18 is used for pathological cross-sequence adaptation experiments.

### Abdominal CT/PET

**FLARE 2025**  
Official: https://openreview.net/group?id=MICCAI.org/2025/Challenge/FLARE

FLARE25 is used for CT-to-PET and PET-to-CT abdominal cross-modality adaptation.

---

## Preprocessing

The preprocessing implementation for the primary CT/MRI experiments is provided in:

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

For the abdominal and cardiac benchmarks, the primary experiments follow the commonly adopted annotation-based ROI preprocessing protocol for direct comparison with previous cross-modality UDA methods. An annotation-free coarse-to-fine preprocessing strategy is additionally evaluated in the Extended Experimental Material.

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

For the FLARE25 PET/CT experiments, annotation-derived ROI localization is not used. PET/CT volumes are processed using reorientation, a fixed central transverse field of view, and spatial normalization before DTCWT decomposition. Because PET intensity statistics differ substantially from CT Hounsfield units and conventional MRI intensities, the CT clipping window `[-125, 275]` should not be directly applied to PET volumes. The DTCWT and subsequent CDFreqNet intervention settings remain unchanged.

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

## Extended Experimental Material

The public **Extended Experimental Material** available from the repository Releases page provides additional analyses, including:

- Annotation-free coarse-to-fine preprocessing experiments.
- Hyperparameter and implementation analyses.
- BraTS18 FLAIR-to-T2 and T2-to-FLAIR pathological cross-sequence adaptation.
- FLARE25 CT-to-PET and PET-to-CT abdominal cross-modality adaptation.
- Multi-source cross-site prostate adaptation.

---
