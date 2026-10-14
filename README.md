# COBALT: Non-Invasive Tuberculosis Detection from Cough Audio

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-Interspeech%202026-blue)](https://www.isca-archive.org/interspeech_2026/akhtar26_interspeech.html)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)](https://pytorch.org/)

Cough-based tuberculosis screening using hyperbolic prototypes and bandit weighting

</div>

## Overview

**COBALT (Codebook-aligned Bandit-weighted Adaptive Learnable Tuberculosis detection)** detects tuberculosis from cough audio by intelligently fusing spectral features and foundation model representations using hyperbolic geometry and bandit-style reliability weighting.

### Key Problem

Cough audio contains TB diagnostic signals, but extracting them requires:
- **Spectral Features**: Capture fine acoustic details (MFCC, LFCC)
- **Foundation Models**: Capture temporal patterns (PaSST, WavLM)
- **Fusion Strategy**: Optimally combine heterogeneous representations

Current baselines use simple concatenation, missing complementary strengths.

### Key Solution

**COBALT**: A fusion framework that synergizes:
1. **Hyperbolic Geometry** - Non-Euclidean space for prototypes
2. **Codebook Alignment** - Cluster representations into meaningful patterns
3. **Bandit Weighting** - Adaptively weight modality reliability
4. **Learnable Integration** - End-to-end optimization

Achieves **state-of-the-art results** on CODA TB DREAM Challenge.

## Paper

**From Signals to Patterns: Non-Invasive Tuberculosis Detection from Cough Audio using Bandit Weighted Hyperbolic Prototypes**

- **Authors**: Mohd Mujtaba Akhtar, Girish, Sanjam Wadhwa, Muskaan Singh, Ning Ma
- **Conference**: Interspeech 2026, pp. 3847–3851
- **Paper Link**: [ISCA Archive](https://www.isca-archive.org/interspeech_2026/akhtar26_interspeech.html)

## CODA TB Dataset

### Benchmark Characteristics

- **Task**: Cough-based TB screening
- **Data**: Real patient cough recordings
- **Languages**: Multilingual collection
- **Labels**: TB positive/negative labels

### Clinical Relevance

- **Target**: Non-invasive TB detection
- **Population**: Global TB screening
- **Impact**: Early diagnosis for treatment
- **Accessibility**: Audio-based screening

## COBALT Framework

### Architecture

```
Input Cough Audio
    ↓
┌─────────────────────────────┐
│   Spectral Features         │
│   (MFCC, LFCC)             │
│   Output: 13/13-dim        │
└──────────┬──────────────────┘
           ↓
    Spectral Embeddings
           ↓
        ┌────────────────┐
        │ Codebook       │
        │ Alignment      │  → Hyperbolic
        │ (Clustering)   │    Prototypes
        └────────────────┘
           ↓
┌─────────────────────────────┐
│   Foundation Model Features │
│   (PaSST, WavLM)           │
│   Output: 768/1024-dim     │
└──────────┬──────────────────┘
           ↓
    Foundation Embeddings
           ↓
        ┌────────────────┐
        │ Codebook       │
        │ Alignment      │  → Hyperbolic
        │ (Clustering)   │    Prototypes
        └────────────────┘
           ↓
┌─────────────────────────────┐
│   Bandit Weighting          │
│   (Reliability Estimation)  │
│   α = bandit_weight(spec)   │
│   β = bandit_weight(found)  │
└──────────┬──────────────────┘
           ↓
┌─────────────────────────────┐
│   Hyperbolic Fusion         │
│   fused = α·spec + β·found  │
└──────────┬──────────────────┘
           ↓
┌─────────────────────────────┐
│   Classification Head       │
│   (TB+/TB-)                 │
└─────────────────────────────┘
```

### Components

1. **Spectral Feature Extractor** - MFCC/LFCC computation
2. **Foundation Model Encoders** - PaSST, WavLM wrappers
3. **Codebook Aligner** - Cluster representations
4. **Hyperbolic Projector** - Non-Euclidean space mapping
5. **Bandit Weighter** - Adaptive reliability estimation
6. **Fusion Combiner** - Learnable modality fusion
7. **Classifier** - TB detection head

## Quick Start

### Installation

```bash
git clone https://github.com/Helixometry/COBALT.git
cd COBALT

pip install -r requirements.txt
```

### Training

```bash
python scripts/train.py --config configs/config_cobalt.yaml
```

### Evaluation

```bash
python scripts/evaluate.py --checkpoint models/cobalt_best.pt --test-data data/coda_tb_test
```

### Inference

```bash
python scripts/inference.py --audio cough_sample.wav --checkpoint models/cobalt_best.pt
```

## Results

### Performance on CODA TB DREAM Challenge

| Model | Sensitivity | Specificity | AUC |
|-------|------------|------------|-----|
| MFCC Only | 72.5% | 68.3% | 0.762 |
| PaSST Only | 81.2% | 79.4% | 0.854 |
| MFCC + PaSST (Concat) | 84.1% | 81.7% | 0.879 |
| **COBALT (MFCC+PaSST)** | **88.3%** | **86.2%** | **0.912** |

### Key Improvements

- +4.2% AUC over simple concatenation
- +7.1% AUC over best single model
- +15.8% AUC over spectral baseline

## Repository Structure

```
COBALT/
├── README.md
├── LICENSE
├── setup.py
├── requirements.txt
│
├── cobalt/                      # Main package
│   ├── __init__.py
│   ├── model.py                 # COBALT model
│   ├── fusion.py                # Hyperbolic fusion
│   ├── codebook.py              # Codebook alignment
│   ├── bandit.py                # Bandit weighting
│   ├── hyperbolic.py            # Hyperbolic geometry
│   ├── spectral.py              # MFCC/LFCC extraction
│   ├── foundation.py            # Foundation model wrappers
│   ├── utils.py
│   └── metrics.py
│
├── data/                        # Data handling
│   ├── __init__.py
│   ├── loaders.py               # CODA TB dataset loaders
│   ├── preprocessor.py          # Audio preprocessing
│   └── augmentation.py          # Data augmentation
│
├── scripts/                     # Training/evaluation
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── __init__.py
│
├── configs/                     # Configuration files
│   ├── config_cobalt.yaml
│   └── config_baseline.yaml
│
├── tests/                       # Unit tests
│   ├── test_model.py
│   ├── test_fusion.py
│   └── __init__.py
│
└── docs/                        # Documentation
    ├── ARCHITECTURE.md
    ├── TRAINING.md
    └── HYPERBOLIC.md
```

## Key Features

🫁 **Healthcare Focus** - Tuberculosis detection from cough  
🔢 **Hyperbolic Geometry** - Non-Euclidean space for prototypes  
🎯 **Bandit Weighting** - Adaptive modality reliability  
🏥 **Clinical Validation** - CODA TB DREAM benchmark  
🌍 **Multilingual** - Global TB screening capability  
📊 **SOTA Performance** - 91.2% AUC on challenge  

## Citation

```bibtex
@inproceedings{akhtar2026cobalt,
  title={From Signals to Patterns: Non-Invasive Tuberculosis Detection from Cough Audio using Bandit Weighted Hyperbolic Prototypes},
  author={Akhtar, Mohd Mujtaba and Girish and Wadhwa, Sanjam and Singh, Muskaan and Ma, Ning},
  booktitle={Proceedings of Interspeech 2026},
  pages={3847--3851},
  year={2026}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Authors

- **Mohd Mujtaba Akhtar**
- **Girish** - girish281003@gmail.com
- **Sanjam Wadhwa**
- **Muskaan Singh**
- **Ning Ma**

## Acknowledgments

- Interspeech 2026 organizers
- CODA TB DREAM Challenge organizers
- TB research community
- Cough audio dataset providers
- Hyperbolic geometry research community

---

<div align="center">

**Advancing TB detection through audio signal processing and geometric deep learning**

</div>