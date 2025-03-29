# Diffusion-Multi-Agent-Meta-Learning

Code repository for the paper [**Dif-MAML: Decentralized Multi-Agent Meta-Learning**](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9669064).

## 🧾 Abstract

The objective of meta-learning is to exploit knowledge obtained from observed tasks to improve adaptation to unseen tasks. Meta-learners are able to generalize better when they are trained with a larger number of observed tasks and with a larger amount of data per task. Given the amount of resources that are needed, it is generally difficult to expect the tasks, their respective data, and the necessary computational capacity to be available at a single central location. It is more natural to encounter situations where these resources are spread across several agents connected by some graph topology.

The formalism of meta-learning is actually well-suited for this decentralized setting, where the learner benefits from information and computational power spread across the agents. Motivated by this observation, we propose a cooperative fully-decentralized multi-agent meta-learning algorithm, referred to as **Diffusion-based MAML (Dif-MAML)**.

Decentralized optimization algorithms are superior to centralized implementations in terms of scalability, robustness, avoidance of communication bottlenecks, and privacy guarantees. The work provides a detailed theoretical analysis to show that the proposed strategy allows a collection of agents to attain agreement at a linear rate and to converge to a stationary point of the aggregate MAML objective even in non-convex environments. Simulation results illustrate the theoretical findings and the superior performance relative to the traditional non-cooperative setting.


## 📚 Citation

🌟 *If you find this code and paper helpful in your research, please consider starring this repository and citing the following paper:*

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


