# DINOv2 JAX

This repository contains a port of FAIR's [DINOv2](https://dinov2.metademolab.com/) to JAX, forked from [kylestach](https://github.com/kylestach/dinov2-jax) and modified to run inference against the pretrained DINO weights.

Use `dino_weights.py` for loading pretrained weights into a ViT-S JAX model (with the same modifications as are made in the DINO paper).

> **Warning**: There are currently some minor discrepancies between the output of the JAX model and the original model. The results are mostly identical, and the difference is likely down to numerical differences in the JAX and pytorch implementations, but there are no guarantees of correctness.

Different from the original repository, this one is supposed to train the model from scratch or fine-tune it on a custom dataset via Jax. The training script is based on my another repository [Jax-Fit](https://github.com/noahzhy/Jax-Fit) which is a general-purpose training script for Jax models and can be easily adapted to train other models in Jax by changing few lines of code.
