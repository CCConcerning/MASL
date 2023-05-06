## Selective Learning for Sample-Efficient Training in Multi-Agent Sparse Reward Tasks
## Requirements
* Python 3.6.9 (Minimum)
* [OpenAI baselines](https://github.com/openai/baselines)
* My [fork] of Multi-agent Particle Environments
* [OpenAI Gym](https://github.com/openai/gym), version: 0.10.5
* [TensorFlow] version: 1.13.1
The versions are just what I used and not necessarily strict requirements.

## Installation
- To install, `cd` into the 'experiments' directory and type `pip install -e .`
- `cd multiagent-particle-envs-master` to install the multiagent particle envs of my repo.

## Run an experiment
```shell
cd experiments
```
All training code is contained within `train.py`. 
Run an experiment on rover exploration:
```shell
python train.py --scenario hunt--max-episode-len 25
```
Run an experiment on resource collection:
```shell
python train.py --scenario simple_spread --max-episode-len 50" for resource collection tasks
```
