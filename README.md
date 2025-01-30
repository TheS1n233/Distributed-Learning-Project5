# Project: "Adaptive LocalSGD: Optimizing Communication Efficiency in Distributed Deep Learning"
## Students: s332145 Andrea Bioddo ; s323024 Yujie Mu 

## Experiment Setup and Training Strategies

This project investigates optimization strategies for distributed deep learning using LocalSGD. The primary goal is to optimize communication efficiency while maintaining model accuracy.

### **1. Centralized Baseline**
- **Objective:** Establish a baseline for comparison.
- **Optimizers Evaluated:** SGDM and AdamW.
- **Hyperparameter Search:**
  - SGDM: LR = 0.01, WD = 0.001, Momentum = 0.9.
  - AdamW: LR = 0.001, WD = 0.1.
- **Results:**
  - SGDM achieved **55.24%** test accuracy.
  - AdamW achieved **52.67%** test accuracy.
  - SGDM showed faster convergence but risked overfitting.

### **2. Large-Batch Training**
- **Objective:** Evaluate large-batch optimizers' scalability.
- **Optimizers Evaluated:** SGDM, LARS, AdamW, LAMB.
- **Experiment Setup:**
  - Batch sizes tested: `{128, 256, 512, 1024, 2048, 4096, 8192}`.
  - Learning rate scaling applied using square root scaling.
  - Cosine learning rate schedule with linear warmup for 5 epochs.
- **Results:**
  - **LARS & LAMB** performed better at larger batch sizes.
  - Accuracy declined significantly beyond batch size **4096**.
  - SGDM and AdamW were less stable at large batch sizes.

### **3. LocalSGD Training**
- **Objective:** Explore local update strategies to reduce communication overhead.
- **Experiment Setup:**
  - Number of workers (K) = `{2, 4, 8}`.
  - Local steps (J) = `{4, 8, 16, 32, 64}`.
  - Total iterations kept constant across experiments.
- **Results:**
  - Best accuracy: **52.86% at K=2, J=32**.
  - Increasing J reduced communication time but degraded convergence.
  - Higher K values introduced synchronization challenges.

### **4. Exploring Advanced Distributed Optimizers (Inner-Outer Loop)**
- **Objective:** Evaluate optimization using a two-level update mechanism.
- **Strategy:**
  - **Inner Loop:** Local updates before synchronization.
  - **Outer Loop:** Synchronization after a predefined number of inner steps.
- **Experiment Setup:**
  - Workers (K) = `{2, 4, 8}`.
  - Local Steps (J) = `{4, 8, 16, 32, 64}`.
  - Fixed Learning Rate = **0.01**.
- **Results:**
  - Best accuracy: **53.82% at K=2, J=32**.
  - Higher J further reduced communication time.
  - Inner-outer loop improved stability over standard LocalSGD.

### **5. Adaptive LocalSGD (Proposed Method)**
- **Objective:** Dynamically adjust local steps (J) based on validation loss.
- **Adaptive Criteria:**
  - Increase J when validation loss stabilizes.
  - Decrease J if oscillations or divergence occur.
- **Experiment Setup:**
  - Workers (K) = `{2, 4, 8}`.
  - Initial J = **16**, dynamically adjusted.
  - Validation loss thresholds: **0.02 (early), 0.005 (mid), 0.001 (late).**
- **Results:**
  - Best accuracy: **53.47% at K=2, J=4**.
  - Up to **84% reduction in communication overhead**.
  - Comparable accuracy to fixed LocalSGD with improved efficiency.

---

## **Main Components and Scripts**

### **Training Scripts**
- `hyperparams_finding.ipynb` - use grid search to find best hyperparameters.
- `centralized_baseline.ipynb` - Runs centralized training with SGDM & AdamW.
- `Large_Batch_Optimezer.ipynb` - Evaluates large-batch optimizers.
- `LocalSGD.ipynb` - Implements standard LocalSGD.
- `SlowMo.ipynb` - Implements two optimizers.
- `Adaptive_J_LocalSGD.ipynb` - Implements Adaptive LocalSGD with dynamic step tuning.

---
## **Experiment Results and Analysis**

### **Performance Summary**
| Method | Best Accuracy | Communication Time Reduction |
|--------|-------------|---------------------------|
| Centralized (SGDM) | **55.24%** | - |
| Large-Batch (LARS/LAMB) | **50.96%** | Moderate |
| LocalSGD (Best J=32) | **52.86%** | Moderate |
| Inner-Outer Loop (Best J=32) | **53.82%** | **Higher efficiency** |
| Adaptive LocalSGD | **53.47%** | **improvement** |

### **Key Findings**
- **LARS & LAMB** improve stability but suffer accuracy drop at **batch size 8192**.
- **Inner-Outer Loop** optimization offers additional stability over LocalSGD.
- **Adaptive LocalSGD** significantly improves communication efficiency without reducing accuracy.

---
