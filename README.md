# Automatic Speech Recognition (ASR)

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About
This repository is an attempt to train an ASR model. Trained model is able to achieve **23%** WER and **10%** CER on `test-clean` dataset, leveraging beam search and language model guidance. Underlying model uses [deepspeech2](https://arxiv.org/abs/1512.02595).
```bash
val_CER_(Argmax): 0.15835317756054224
val_WER_(Argmax): 0.44245870354749056
val_CER_(BeamSearchLM): 0.1086015791507941
val_WER_(BeamSearchLM): 0.2379445747889793
```
Follow the steps desribed in "How to use" section to run inference on the best model to reproduce stated results, or run training to create the model with the same performance.

[Link to wandb artifcats](https://wandb.ai/professor322/asr_model/workspace)

Full report can be found [here](ASR_Report.pdf)

## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```
3. Also make sure that `gzip` utility is installed

## How To Use

To train a model, run the following command:

```bash
python3 train.py -cn=deepspeech_char_colab.yaml trainer.save_dir="saved"
```

To reproduce the stated results

To download model that achieves stated result use this command
```bash
gdown https://drive.google.com/uc\?id\=1-VTff8NcIqxg7wVHARX7GxR-ix92SUVp
```

To run inference (evaluate the model or save predictions):

```bash
python3 python3 inference.py -cn=inference.yaml inferencer.from_pretrained=<path_to_downloaded_model>
```

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
