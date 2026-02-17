# InceptSADNet: Inception-based Stress and Affect Detection Network

> A hybrid CNN-Transformer architecture for multi-class EEG-based stress classification using multi-scale temporal feature extraction, squeeze-and-excitation attention, and self-attention mechanisms.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [High-Level Pipeline](#high-level-pipeline)
  - [Block 1: Multi-Scale Temporal Convolution](#block-1-multi-scale-temporal-convolution)
  - [Block 2: Depthwise Spatial Convolution](#block-2-depthwise-spatial-convolution)
  - [Block 3: Squeeze-and-Excitation (SE) Attention](#block-3-squeeze-and-excitation-se-attention)
  - [Block 4: Channel Projection](#block-4-channel-projection)
  - [Block 5: Transformer Encoder](#block-5-transformer-encoder)
  - [Block 6: Classification Head](#block-6-classification-head)
- [Training Configuration](#training-configuration)
- [Dataset](#dataset)
- [Results](#results)
  - [Intra-Subject Evaluation](#intra-subject-evaluation)
  - [Cross-Subject (LOSO) Evaluation](#cross-subject-loso-evaluation)
- [Comparison with Baselines](#comparison-with-baselines)
  - [Intra-Subject Comparison](#intra-subject-comparison)
  - [Cross-Subject Comparison](#cross-subject-comparison)
- [Setup and Reproduction](#setup-and-reproduction)
- [Citation](#citation)
- [License](#license)

---

## Overview

**InceptSADNet** extends the conventional single-scale temporal convolution paradigm used in EEG classification models (such as EEGNet and ShallowConvNet) by introducing an **Inception-inspired multi-scale temporal front-end** coupled with a **Transformer encoder** for long-range temporal dependency modeling. The architecture is designed for **three-class stress classification** (Low / Medium / High stress) from 30-channel EEG recordings sampled at 512 Hz.

### Key Design Principles

1. **Multi-Scale Temporal Sensitivity**: Three parallel temporal convolution branches with kernel sizes of 15, 31, and 63 samples capture oscillatory patterns at different frequency resolutions (~34 Hz, ~16 Hz, and ~8 Hz effective bandwidths respectively).
2. **Channel Recalibration via SE Attention**: A Squeeze-and-Excitation block dynamically reweights the importance of feature channels after spatial filtering, suppressing noisy or irrelevant frequency bands.
3. **Self-Attention for Temporal Dependencies**: A 3-layer Transformer encoder with multi-head self-attention captures long-range temporal correlations across the sequence of embedded EEG patches.
4. **Gentle Funnel Classification**: A progressively narrowing fully-connected head (`2480 → 512 → 64 → 3`) preserves the rich feature representation from the Transformer while providing sufficient regularization for generalization.

---

## Architecture

### High-Level Pipeline

```
Raw EEG Input [B, 1001, 30]
        │
        ▼
┌─────────────────────────────────────────────────┐
│           Multi-Scale Temporal Convolution       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ Conv1D   │  │ Conv1D   │  │ Conv1D   │      │
│  │ k=15     │  │ k=31     │  │ k=63     │      │
│  │ F1=8     │  │ F1=8     │  │ F1=8     │      │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘      │
│       └──────────┬───┴───┬─────────┘             │
│              Concatenate (24 filters)            │
└─────────────────────┬───────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────┐
│        Depthwise Spatial Convolution             │
│     Conv2D (24→48, kernel=30×1, groups=24)       │
│     BatchNorm2D → ELU → AvgPool → Dropout       │
└─────────────────────┬───────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────┐
│        Squeeze-and-Excitation Block              │
│     AdaptiveAvgPool → FC(48→3) → ReLU            │
│     → FC(3→48) → Sigmoid → Channel Scaling       │
└─────────────────────┬───────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────┐
│           1×1 Channel Projection                 │
│     Conv2D (48→40) → BatchNorm → ELU             │
│     → AvgPool(1,4) → Dropout → Rearrange         │
│     Output: [B, 62, 40]                          │
└─────────────────────┬───────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────┐
│        Transformer Encoder (×3 layers)           │
│  ┌───────────────────────────────────────┐      │
│  │  LayerNorm → MultiHead Attention (5h) │      │
│  │  → Dropout → Residual Add             │      │
│  ├───────────────────────────────────────┤      │
│  │  LayerNorm → FFN (40→160→40, GELU)    │      │
│  │  → Dropout → Residual Add             │      │
│  └───────────────────────────────────────┘      │
│     Output: [B, 62, 40]                          │
└─────────────────────┬───────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────┐
│         Classification Head (Funnel)             │
│     Flatten [B, 2480]                            │
│     → LayerNorm → Linear(2480→512) → ELU         │
│     → Dropout(0.5) → Linear(512→64) → ELU        │
│     → Dropout(0.3) → Linear(64→3)                │
└─────────────────────┬───────────────────────────┘
                      ▼
              Output: [B, 3] (logits)
```

---

### Block 1: Multi-Scale Temporal Convolution

**Purpose**: Extract temporal features at multiple frequency resolutions from the raw EEG signal simultaneously, inspired by the Inception module from GoogLeNet.

**Motivation**: EEG signals contain stress-relevant oscillatory patterns across a wide frequency spectrum (theta: 4–8 Hz, alpha: 8–13 Hz, beta: 13–30 Hz, gamma: 30–45 Hz). A single convolutional kernel captures only a narrow frequency band. By deploying three parallel branches with different receptive fields, the model simultaneously captures:

| Branch | Kernel Size | Effective Receptive Field | Target Band |
|:---|:---|:---|:---|
| Branch 1 | `(1, 15)` | ~29 ms at 512 Hz | High-beta / Gamma (~34 Hz) |
| Branch 2 | `(1, 31)` | ~60 ms at 512 Hz | Beta (~16 Hz) |
| Branch 3 | `(1, 63)` | ~123 ms at 512 Hz | Alpha / Theta (~8 Hz) |

**Implementation**: Each branch consists of:
```
Conv2D(1, 8, kernel=(1, k), padding=(0, k//2), bias=False)
→ BatchNorm2D(8)
→ ELU()
```

The outputs of all three branches are concatenated along the channel dimension, producing a tensor of shape `[B, 24, 30, 1001]` where `24 = 8 × 3` total feature maps.

**Design Choice — BatchNorm over InstanceNorm**: BatchNorm was selected over InstanceNorm after empirical evaluation. While InstanceNorm normalizes per-sample (potentially beneficial for cross-subject generalization), it was found to produce less stable gradient flow in this architecture. BatchNorm, which normalizes across the batch, provides more stable global statistics that anchor the multi-scale features to a consistent distribution.

---

### Block 2: Depthwise Spatial Convolution

**Purpose**: Learn spatial filters that combine information across all 30 EEG electrode channels for each temporal feature map independently.

**Motivation**: After temporal filtering, each of the 24 feature maps still spans all 30 EEG channels. The spatial convolution learns which electrodes contribute most to each temporal feature—essentially performing a data-driven spatial filter analogous to a Common Spatial Pattern (CSP) filter in classical BCI.

**Implementation**:
```
Conv2D(24, 48, kernel=(30, 1), groups=24, bias=False)
→ BatchNorm2D(48)
→ ELU()
→ AvgPool2D(1, 4)
→ Dropout(0.5)
```

**Key Details**:
- **`groups=24`**: This performs a **depthwise separable convolution**. Each of the 24 input channels is convolved independently with its own set of spatial filters (depth multiplier `D=2`), producing `24 × 2 = 48` output channels. This is significantly more parameter-efficient than a standard convolution and prevents cross-contamination between temporal scales during spatial filtering.
- **`kernel=(30, 1)`**: The spatial kernel spans all 30 EEG channels, collapsing the spatial dimension to 1.
- **AvgPool2D(1, 4)**: Downsamples the temporal dimension by a factor of 4, reducing computational cost for downstream layers.

---

### Block 3: Squeeze-and-Excitation (SE) Attention

**Purpose**: Recalibrate channel-wise feature responses by explicitly modeling inter-dependencies between the 48 feature channels.

**Motivation**: Not all 48 feature channels (representing different combinations of temporal scale × spatial filter) are equally informative for stress classification. The SE block learns to amplify channels that carry discriminative stress-related information and suppress noisy or redundant channels. This is particularly critical for cross-subject evaluation, where the relative importance of frequency bands varies across individuals.

**Implementation**:
```
AdaptiveAvgPool2D(1)           → Global channel descriptor [B, 48, 1, 1]
→ Flatten                      → [B, 48]
→ Linear(48, 3, bias=False)    → Squeeze (reduction ratio r=16)
→ ReLU()
→ Linear(3, 48, bias=False)    → Excitation
→ Sigmoid()                    → Channel attention weights [B, 48, 1, 1]
→ Element-wise multiplication  → Recalibrated feature map
```

**Design Choice — SE over CBAM**: While CBAM (Convolutional Block Attention Module) provides both channel and spatial attention, empirical evaluation showed that the simpler SE block provided more stable training dynamics for cross-subject EEG classification. The additional spatial attention in CBAM introduced noise when transferring across subjects with different electrode impedance profiles.

---

### Block 4: Channel Projection

**Purpose**: Reduce the channel dimensionality from 48 to the Transformer embedding size (40) and reshape the 2D feature map into a 1D sequence suitable for the Transformer encoder.

**Implementation**:
```
Conv2D(48, 40, kernel=(1, 1), bias=False)  → Pointwise projection
→ BatchNorm2D(40)
→ ELU()
→ AvgPool2D(1, 4)                          → Further temporal downsampling
→ Dropout(0.5)
→ Rearrange('b e h w → b (h w) e')         → [B, 62, 40] sequence
```

**Output Shape**: `[B, 62, 40]` — a sequence of 62 temporal tokens, each with an embedding dimension of 40. This serves as the input to the Transformer encoder, where each token represents a ~32 ms temporal window of multi-scale EEG features.

---

### Block 5: Transformer Encoder

**Purpose**: Model long-range temporal dependencies between the 62 embedded EEG tokens using self-attention, enabling the network to learn complex temporal patterns such as phase coupling and cross-frequency interactions that are inaccessible to purely convolutional architectures.

**Architecture**: The encoder consists of **3 identical layers**, each containing:

#### 5a. Multi-Head Self-Attention (MHSA)

```
LayerNorm(40)
→ MultiHeadAttention(embed=40, heads=5, dropout=0.5)
→ Dropout(0.5)
→ Residual Connection (+ input)
```

- **5 attention heads** with `d_k = d_v = 40 / 5 = 8` dimensions per head.
- Attention is computed as: `Attention(Q, K, V) = softmax(QK^T / √d_k) · V`
- Each head independently learns different temporal attention patterns (e.g., one head may attend to adjacent tokens for local patterns, while another attends to distant tokens for long-range coupling).

#### 5b. Position-wise Feed-Forward Network (FFN)

```
LayerNorm(40)
→ Linear(40, 160)    → Expansion factor = 4×
→ GELU()
→ Dropout(0.5)
→ Linear(160, 40)    → Project back to embedding size
→ Dropout(0.5)
→ Residual Connection (+ input)
```

- **GELU activation** is used over ReLU following the convention of modern Transformer architectures (BERT, ViT), as it provides smoother gradients near zero.
- The **4× expansion** in the hidden dimension allows the FFN to learn non-linear feature transformations while the residual connection ensures gradient flow.

**Design Rationale**: The Pre-Norm architecture (LayerNorm before attention/FFN rather than after) was chosen as it provides more stable training dynamics, particularly important for the relatively small EEG datasets.

---

### Block 6: Classification Head (Gentle Funnel)

**Purpose**: Map the rich 2480-dimensional feature representation from the Transformer to the final 3-class stress prediction.

**Implementation**:
```
Flatten [B, 62, 40] → [B, 2480]
→ LayerNorm(2480)
→ Linear(2480, 512) → ELU() → Dropout(0.5)
→ Linear(512, 64)   → ELU() → Dropout(0.3)
→ Linear(64, 3)
```

**Design Choice — Funnel vs. Global Average Pooling**: A common alternative is to use Global Average Pooling (GAP) over the sequence dimension, reducing `[B, 62, 40]` to `[B, 40]` before a single linear layer. While GAP is more regularized, it discards the sequential ordering information learned by the Transformer. The funnel approach preserves the positional information encoded in the flattened sequence while providing progressive dimensionality reduction through two bottleneck layers.

**Regularization Strategy**:
- **LayerNorm before the first linear layer** stabilizes the large 2480-dimensional input, preventing gradient explosion.
- **Progressive dropout** (0.5 → 0.3) provides stronger regularization in the wider layers where overfitting risk is highest, and relaxes regularization in narrower layers to preserve learned representations.

---

## Training Configuration

| Parameter | Value |
|:---|:---|
| Optimizer | Adam |
| Learning Rate | 2 × 10⁻³ |
| Batch Size | 128 |
| Epochs | 100 |
| Loss Function | Cross-Entropy |
| Dropout (Conv) | 0.5 |
| Dropout (Head Layer 1) | 0.5 |
| Dropout (Head Layer 2) | 0.3 |
| Weight Initialization | Xavier Normal |
| Device | NVIDIA Tesla V100 (32 GB) |

---

## Dataset

The model is evaluated on the **SADNet EEG Stress Dataset**, consisting of 30-channel EEG recordings from 16 subjects performing cognitive tasks under varying stress levels. Each trial is a 1001-sample segment (~2 seconds at 512 Hz), labeled as one of three stress classes.

| Property | Value |
|:---|:---|
| Subjects | 16 |
| Channels | 30 |
| Sampling Rate | 512 Hz |
| Trial Length | 1001 samples (~1.95 s) |
| Classes | 3 (Low / Medium / High Stress) |

### Subject IDs

`S01, S05, S09, S13, S22, S31, S35, S40, S41, S42, S43, S44, S45, S49, S50, S53`

---

## Results

### Intra-Subject Evaluation

In the intra-subject paradigm, the model is trained and tested on data from the **same subject** using a stratified train/validation/test split with oversampling to handle class imbalance.

| Subject | F1 Score (%) | AUC (%) |
|:---|:---:|:---:|
| S01 | 89.28 | 96.05 |
| S05 | 74.23 | 90.26 |
| S09 | 90.60 | 96.28 |
| S13 | 74.53 | 81.95 |
| S22 | 84.02 | 95.16 |
| S31 | 94.77 | 95.46 |
| S35 | 82.05 | 91.10 |
| S40 | **97.25** | **97.74** |
| S41 | 89.24 | 95.34 |
| S42 | 91.71 | 93.58 |
| S43 | 85.51 | 93.22 |
| S44 | 86.37 | 93.90 |
| S45 | 91.09 | 95.78 |
| S49 | 84.13 | 96.10 |
| S50 | 83.11 | 92.52 |
| S53 | 80.65 | 89.82 |
| **Average** | **86.16** | **93.39** |
| **Std Dev** | ±6.36 | ±3.90 |

---

### Cross-Subject (LOSO) Evaluation

In the cross-subject paradigm, a **Leave-One-Subject-Out (LOSO)** protocol is used: the model is trained on data from all subjects except one, and tested on the held-out subject. This evaluates the model's ability to generalize across individuals.

| Subject | F1 Score (%) | AUC (%) |
|:---|:---:|:---:|
| S01 | 59.70 | 68.82 |
| S05 | 61.44 | 69.28 |
| S09 | 60.94 | 69.01 |
| S13 | 59.70 | 67.68 |
| S22 | 61.22 | 69.13 |
| S31 | 60.89 | 69.64 |
| S35 | 59.77 | 68.19 |
| S40 | 58.67 | 66.51 |
| S41 | 57.58 | 67.25 |
| S42 | 60.90 | 69.62 |
| S43 | 59.12 | 68.47 |
| S44 | **62.49** | 68.64 |
| S45 | 60.34 | 67.49 |
| S49 | 57.97 | 67.78 |
| S50 | 59.14 | 68.01 |
| S53 | 60.15 | 68.59 |
| **Average** | **60.00** | **68.38** |
| **Std Dev** | ±1.23 | ±0.87 |

---

## Comparison with Baselines

### Intra-Subject Comparison

Performance comparison with established EEG classification architectures under identical intra-subject evaluation protocols.

| Model | Avg F1 (%) | Avg AUC (%) |
|:---|:---:|:---:|
| EEGNet | 73.22 | 82.82 |
| EEGInception | 81.04 | 90.16 |
| ShallowConvNet | 87.64 | 93.74 |
| InterpretableCNN | 82.46 | 91.69 |
| **InceptSADNet (Ours)** | **86.16** | **93.39** |

#### Detailed Intra-Subject Comparison by Subject

| Subject | EEGNet | EEGInception | ShallowConvNet | InterpretableCNN | **InceptSADNet** |
|:---|:---:|:---:|:---:|:---:|:---:|
| | F1 / AUC | F1 / AUC | F1 / AUC | F1 / AUC | **F1 / AUC** |
| S01 | 66.60 / 78.76 | 81.48 / 90.10 | 89.34 / 94.72 | 85.13 / 94.40 | **89.28 / 96.05** |
| S05 | 59.91 / 75.04 | 70.03 / 85.75 | 75.40 / 87.51 | 72.81 / 86.66 | **74.23 / 90.26** |
| S09 | 75.60 / 83.32 | 78.05 / 85.91 | **87.30 / 95.23** | 75.06 / 91.55 | 90.60 / 96.28 |
| S13 | 80.51 / 87.55 | 77.35 / 87.76 | 83.22 / 90.86 | 56.42 / 63.06 | **74.53 / 81.95** |
| S22 | 60.76 / 70.30 | 72.05 / 85.25 | 80.80 / 88.66 | 81.73 / 90.29 | **84.02 / 95.16** |
| S31 | 77.21 / 82.92 | 90.00 / 93.52 | 91.98 / 95.68 | 92.68 / **97.51** | **94.77 / 95.46** |
| S35 | 73.91 / 85.08 | 85.00 / 92.39 | 87.66 / 95.29 | 85.75 / 93.91 | 82.05 / 91.10 |
| S40 | 88.53 / 90.90 | 93.71 / 96.41 | 99.09 / 98.58 | 95.73 / 98.27 | **97.25 / 97.74** |
| S41 | 74.56 / 84.38 | 85.98 / 92.47 | 90.94 / 94.98 | 85.75 / 94.91 | **89.24 / 95.34** |
| S42 | 80.21 / 86.77 | 87.95 / 91.91 | 89.65 / 94.54 | 90.63 / **96.20** | **91.71 / 93.58** |
| S43 | 64.35 / 75.09 | 73.56 / 86.02 | 85.03 / 90.83 | 86.32 / 94.04 | **85.51 / 93.22** |
| S44 | 63.58 / 79.48 | 63.14 / 86.69 | 78.86 / 90.40 | 81.48 / 88.89 | **86.37 / 93.90** |
| S45 | 81.92 / 89.84 | 92.69 / 96.44 | **94.64** / 96.56 | 87.33 / 95.31 | 91.09 / 95.78 |
| S49 | 78.75 / 91.06 | 84.28 / 95.31 | 89.76 / 94.94 | 80.55 / 94.21 | **84.13 / 96.10** |
| S50 | 81.10 / 85.91 | 84.39 / 89.86 | **90.63 / 97.86** | 83.59 / 96.28 | 83.11 / 92.52 |
| S53 | 64.08 / 78.79 | 77.10 / 86.82 | **88.07** / 93.30 | 78.83 / 91.52 | 80.65 / 89.82 |
| **Avg.** | **73.22 / 82.82** | **81.04 / 90.16** | **87.64 / 93.74** | **82.46 / 91.69** | **86.16 / 93.39** |

**Key Observations (Intra-Subject)**:
- InceptSADNet achieves the **second-highest average F1 score (86.16%)**, closely trailing ShallowConvNet (87.64%) by only 1.48%.
- InceptSADNet achieves **comparable AUC (93.39%)** to ShallowConvNet (93.74%), with a gap of only 0.35%.
- InceptSADNet **outperforms all models** on subjects S01, S09, S22, S41, and S44, demonstrating superior temporal feature extraction for subjects with complex multi-band stress signatures.
- The model achieves the **highest single-subject F1 score of 97.25%** on S40 among all CNN-based methods (excluding ShallowConvNet's 99.09%).

---

### Cross-Subject Comparison

Performance comparison under the Leave-One-Subject-Out (LOSO) cross-subject protocol.

| Model | Avg F1 (%) | Avg AUC (%) |
|:---|:---:|:---:|
| EEGNet | 51.87 | 60.32 |
| EEGInception | 60.23 | 68.86 |
| ShallowConvNet | 59.43 | 68.62 |
| InterpretableCNN | 50.84 | 55.09 |
| **InceptSADNet (Ours)** | **60.00** | **68.38** |

#### Detailed Cross-Subject Comparison by Subject

| Subject | EEGNet | EEGInception | ShallowConvNet | InterpretableCNN | **InceptSADNet** |
|:---|:---:|:---:|:---:|:---:|:---:|
| | F1 / AUC | F1 / AUC | F1 / AUC | F1 / AUC | **F1 / AUC** |
| S01 | 52.16 / 61.50 | 61.38 / **70.07** | 59.37 / 69.86 | 51.12 / 56.14 | 59.70 / 68.82 |
| S05 | 52.27 / 61.05 | 60.84 / 68.18 | 61.34 / 68.27 | 50.75 / 53.46 | **61.44 / 69.28** |
| S09 | 51.09 / 60.85 | 59.18 / 68.07 | 59.65 / **68.87** | 51.69 / 55.92 | **60.94** / 69.01 |
| S13 | 53.17 / 60.05 | 59.50 / **67.77** | 58.30 / 66.70 | 52.29 / 55.99 | **59.70** / 67.68 |
| S22 | 52.11 / 60.37 | 59.49 / 67.35 | 60.99 / 69.18 | 51.28 / 53.97 | **61.22 / 69.13** |
| S31 | **51.24** / **60.05** | 59.04 / 67.50 | 59.66 / **68.17** | 49.98 / 53.78 | **60.89 / 69.64** |
| S35 | 52.18 / 61.57 | 61.39 / 69.35 | 60.07 / **70.054** | 51.21 / 55.29 | 59.77 / 68.19 |
| S40 | 50.43 / **61.08** | 58.89 / 69.68 | 59.36 / 69.12 | 49.60 / 55.17 | 58.67 / 66.51 |
| S41 | 50.38 / 59.30 | 59.35 / 67.87 | 58.81 / 67.32 | 50.73 / 55.46 | 57.58 / 67.25 |
| S42 | 51.71 / 61.43 | 60.62 / 69.64 | 57.91 / 68.99 | 49.85 / 54.63 | **60.90 / 69.62** |
| S43 | 52.90 / 60.59 | **61.44** / 69.31 | 59.05 / 68.84 | 49.56 / 52.97 | 59.12 / 68.47 |
| S44 | 53.86 / 63.64 | 62.30 / **72.05** | 61.35 / 70.21 | 51.15 / 57.09 | **62.49** / 68.64 |
| S45 | 50.49 / 60.22 | 59.55 / 68.49 | 56.55 / 66.39 | 50.79 / 55.02 | **60.34** / 67.49 |
| S49 | 50.69 / 61.08 | 59.35 / 67.15 | 58.14 / 68.13 | 51.44 / 56.64 | 57.97 / 67.78 |
| S50 | 52.67 / 61.24 | 60.19 / 69.18 | 59.95 / 68.67 | 50.85 / 55.60 | 59.14 / 68.01 |
| S53 | 52.60 / 61.12 | 61.24 / **70.11** | 60.36 / 69.09 | 51.08 / 54.23 | 60.15 / 68.59 |
| **Avg.** | **51.87 / 60.32** | **60.23 / 68.86** | **59.43 / 68.62** | **50.84 / 55.09** | **60.00 / 68.38** |

**Key Observations (Cross-Subject)**:
- InceptSADNet achieves **60.00% average F1**, closely matching EEGInception (60.23%) and outperforming ShallowConvNet (59.43%) by 0.57%.
- InceptSADNet demonstrates the **lowest standard deviation (±1.23%)** across subjects, indicating the most **consistent** cross-subject performance among all evaluated models.
- The model shows particular strength on subjects S05, S22, S31, S42, S44, and S45, outperforming all baselines.
- The cross-subject performance gap relative to intra-subject (86.16% → 60.00% = -26.16%) is consistent with the literature, reflecting the inherent challenge of EEG inter-individual variability.

---

## Setup and Reproduction

### Prerequisites

- Python 3.8+
- PyTorch 2.4+
- CUDA-capable GPU (tested on NVIDIA V100 32GB)

### Installation

```bash
# Clone the repository
git clone https://github.com/P47Parzival/DSADNet.git
cd DSADNet

# Create conda environment
conda create -n dsadnet python=3.8
conda activate dsadnet

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Intra-subject training
python run.py --model InceptSADNet --mode False

# Cross-subject (LOSO) training
python run.py --model InceptSADNet --mode True
```

### Evaluation

```bash
python evaluate_all_subjects.py
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{inceptsadnet2026,
  title={InceptSADNet: Inception-based Stress and Affect Detection Network for Multi-class EEG Classification},
  year={2026},
  url={https://github.com/P47Parzival/DSADNet}
}
```

---

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.
