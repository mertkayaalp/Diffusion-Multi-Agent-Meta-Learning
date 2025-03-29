# Diffusion-based Multi-Agent Meta-Learning

Code repository for the paper [**Dif-MAML: Decentralized Multi-Agent Meta-Learning**](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9669064).

## Introduction

This repository implements Dif-MAML (Diffusion-based Model-Agnostic Meta-Learning). Dif-MAML is a **decentralized multi-agent meta-learning algorithm**. [Classic MAML algorithm](https://proceedings.mlr.press/v70/finn17a.html) requires a single agent to train on all tasks. In contrast, Dif-MAML enables multiple agents, each having their own tasks and data, to cooperatively learn a shared initialization. Each agent computes local updates using its tasks, then averages those updates with neighbors. 

## Tasks and Datasets

The experiments are divided into two categories. 

### Regression
- **Sine Function Regression:** Agents learn to predict sine functions of varying amplitudes and phases. Each agent sees different amplitude and phase ranges but cooperates via Dif-MAML algorithm to learn a shared launch model.

### Few-Shot Classification
- **Omniglot:** Character recognition tasks in 1- or 5-shot scenarios. Each agent has access to a subset of characters. 
- **MiniImageNet:** ImageNet tasks (5-way, 1-shot or 5-shot). Each agent has access to a subset of classes and images.

## 📚 Citation

🌟 *If you find this code or approach helpful in your research, please consider starring this repository and citing the following paper:*

```bibtex
@ARTICLE{kayaalp2022,
  author={Kayaalp, Mert and Vlaski, Stefan and Sayed, Ali H.},
  journal={IEEE Open Journal of Signal Processing}, 
  title={Dif-MAML: Decentralized Multi-Agent Meta-Learning}, 
  year={2022},
  volume={3},
  number={},
  pages={71-93},
  doi={10.1109/OJSP.2021.3140000}}


