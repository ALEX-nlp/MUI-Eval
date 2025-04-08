# Revisiting LLM Evaluation through Mechanism
Interpretability: a New Metric and Model Utility Law

The author XXX, XXX

---

[![arXiv](https://img.shields.io/badge/arXiv-2410.08261-d63333.svg?logo=arxiv)]()
[![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)]()

This repository contains the implementation code and associated data for our newly proposed metric, the Model Utilization Index (MUI). This metric is currently derived using Neuron-Based Interpretation and Sparse Activation Encoder (SAE)-based Interpretation methodologies

## 🎉 News 

- [x] [2025-04-010] 🎉 Exciting Release! We’ve just published our paper titled "Revisiting LLM Evaluation through Mechanism Interpretability: A New Metric and Model Utility Law". Check out our repository for the corresponding analysis code!

## Brief Introduction

The Model Utilization Index (MUI) quantifies the extent to which a model leverages its capabilities to complete tasks. The core concept behind MUI is that evaluating an LLM’s overall ability involves more than just measuring its task performance; it also requires analyzing the effort expended to achieve those outcomes. Our extensive experiments reveal an inverse relationship between MUI and performance, suggesting a common trend observed in popular LLMs. We have termed this phenomenon the "Utility Law."

<p align="center" width="100%">
<a target="_blank"><img src="figures/Utility_Law.png" alt="MUI" style="width: 90%; min-width: 200px; display: block; margin: auto;"></a>
</p>

## 🛠️ Prerequisites

### Step 1: Clone the repository

```bash
git clone 
cd MUI
```

### Step 2: Create virtual environment

```bash
conda create --name meissonic python
conda activate meissonic
pip install -r requirements.txt
```

### Step 3: Install You Target Models 

```bash
git clone https://github.com/huggingface/diffusers.git
cd diffusers
pip install -e .
```

## 💡 Inference Usage

## Contact

For any questions or feedback, feel free to contact [Jiahao Ying](jhying.2022@phdcs.smu.edu.sg) and [Yixin Cao](caoyixin2011@gmail.com).


## 📚 Citation

If you find this work helpful, please consider citing:

```bibtex

```
