# Cognitive Training Simulation

This repository contains the code used to run the simulations and generate the plots for the paper [Cognitive Training for Language Models: Towards General Capabilities via Cross-Entropy Games](https://arxiv.org/abs/2603.22479).

## Install


```bash
uv sync
```

## Run

```bash
uv run python simulation.py
```

The plots are generated in `plots/`.

## Changing the Simulation

To run the simulation with different settings, change the variables at the top of [simulation.py](./simulation.py).
