# Personalizing Low-rank Bayesian Neural Networks via Federated Learning

This repository provides the **source code** for the paper:

> *Personalizing Low-rank Bayesian Neural Networks via Federated Learning*  

---

## 📌 Overview
Reliable uncertainty quantification is crucial for real-world decision-making, especially in **personalized federated learning (PFL)** where each client typically has limited and heterogeneous local data. While Bayesian PFL (BPFL) methods improve calibration, they are often computationally and memory intensive since they must track the variances of all parameters.

To address these challenges, we propose **LR-BPFL**, a novel method that combines:
- A **global deterministic model** shared across clients  
- **Personalized low-rank Bayesian corrections** with adaptive rank selection to reflect each client’s inherent uncertainty  

Our experiments across diverse datasets show that **LR-BPFL** achieves superior calibration and accuracy while significantly reducing computational and memory overhead.


## 🚀 Getting Started
Clone this repository:
```bash
git clone git@github.com:BernieZhang15/PFL_framework.git
cd PFL_framework


