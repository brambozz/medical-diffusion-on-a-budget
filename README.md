# Medical diffusion on a budget

Repository for the paper 'Medical diffusion on a budget: textual inversion for medical image generation'.

## Textual Inversion

Textual Inversion and image generation was performed with the [AUTOMATIC1111 web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui).
Specifically, the version of the repository at [commit `d050bb7`](https://github.com/AUTOMATIC1111/stable-diffusion-webui/tree/d050bb78636236338440768f871a0f2bb9f277f6) was used.

To start generating with the embeddings, follow the installation instructions there and use the Stable Diffusion 2.0 [checkpoint](https://huggingface.co/stabilityai/stable-diffusion-2), specifically `512-base-ema.ckpt`.

## Trained embeddings

All trained embeddings are included in the `embeddings` folder.

## Classifier training

The environment used to train the binary classifiers can be recreated from `requirements.txt`.

Models can be trained with the `train.py` script, which is based on [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) [Flash](https://github.com/Lightning-Universe/lightning-flash) and [Hydra](https://github.com/facebookresearch/hydra).

Default configuration can be set in `conf/config.yaml`.

## FID scores

FID scores were calculated with `evaluate_generation_quality.py`, e.g.

```sh
python evaluate_generation_quality.py --generated_path /path/to/generated/images --reference_path /path/to/reference/images
```

## StyleGAN3 baseline

For the StyleGAN3 baseline used in the paper, we refer to the
[original repository](https://github.com/NVlabs/stylegan3) for details on installation and requirements.

The following command was used to train:

```sh
train.py --cfg=stylegan3-t --data=/path/to/train/set --gpus=1 --batch=4 --gamma=8 --mirror=1 --kimg=100 --resume=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-afhqv2-512x512.pkl --snap=25 --tick=1 --mbstd-group 1 --metrics none
```

## Poster template

Feel free to use the poster template, made with [Quarto](https://quarto.org).
To preview/render the poster:

```
cd poster
quarto preview/render poster.qmd
```
