# COBALT Architecture

## Overview

COBALT (Codebook-aligned Bandit-weighted Adaptive Learnable Tuberculosis detection) detects TB from cough audio using hyperbolic geometry and bandit-style reliability weighting.

## Key Components

### 1. Spectral Features
- MFCC/LFCC extraction from cough audio
- Captures fine acoustic details
- Low-dimensional representation

### 2. Foundation Models
- PaSST or WavLM encoders
- Capture temporal/semantic patterns
- High-dimensional embeddings

### 3. Codebook Alignment
- Cluster representations into prototypes
- Create discrete codebook
- Project into hyperbolic space

### 4. Hyperbolic Geometry
- Non-Euclidean space for TB patterns
- Better representation of hierarchical structures
- Natural distance metrics for health states

### 5. Bandit Weighting
- Estimate reliability of each modality
- Adaptive weight assignment
- Context-dependent fusion

## Architecture

```
Input → Spectral Features → Codebook → Hyperbolic → Bandit → Fusion → Classification
        (MFCC/LFCC)        Alignment   Projection   Weight
         ↓
        Foundation → Codebook → Hyperbolic 
        Features    Alignment   Projection
        (PaSST)
```

## Mathematical Formulation

- **Spectral**: s = MFCC(cough)
- **Foundation**: f = PaSST(cough)
- **Codebook align**: cs = argmin ||s - prototype_s||
- **Bandit weight**: w_s = confidence(s), w_f = confidence(f)
- **Fusion**: z = (w_s * s + w_f * f) / (w_s + w_f)
- **Output**: logits = classifier(z)

## Performance

- Sensitivity: 88.3%
- Specificity: 86.2%
- AUC: 0.912
