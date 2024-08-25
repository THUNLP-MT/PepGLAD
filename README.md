# PepGLAD: Full-Atom Peptide Design with Geometric Latent Diffusion

TODO: cover

## Quick Links

## Setup

### Environment

The conda environment can be constructed with the configuration `env.yaml`:

```bash
conda env create -f env.yaml
```

The codes are tested with cuda version `11.7` and pytorch version `1.13.1`.

Don't forget to activate the environment before running the codes:

```bash
conda activate PepGLAD
```


### Datasets

1. Download

The datasets are uploaded to Zenodo at [this url](https://zenodo.org/records/13358011). You can download them as follows:

```bash
mkdir datasets  # all datasets will be put into this directory
wget https://zenodo.org/records/13358011/files/train_valid.tar.gz?download=1 -O ./datasets/train_valid.tar.gz   # training/validation
wget https://zenodo.org/records/13358011/files/LNR.tar.gz?download=1 -O ./datasets/LNR.tar.gz   # test set
wget https://zenodo.org/records/13358011/files/ProtFrag.tar.gz?download=1 -O ./datasets/ProtFrag.tar.gz     # augmentation dataset
```

2. Decompresss

```bash
tar zxvf ./datasets/train_valid.tar.gz -C ./datasets
tar zxvf ./datasets/LNR.tar.gz -C ./datasets
tar zxvf ./datasets/ProtFrag.tar.gz -C ./datasets
```

3. Process

### (Optional) Trained Weights

TODO: in release


## Usage

### Peptide Sequence-Structure Co-Design

### Peptide Binding Structure Prediction


## Reproduction of Paper Experiments

### Codesign

Train autoencoder

Train latent diffusion

Inference

Evaluation

### Binding Conformation Generation

Train autoencoder

Train latent diffusion

Inference

Evaluation

## Contact

Thank you for your interest in our work!

Please feel free to ask about any questions about the algorithms, codes, as well as problems encountered in running them so that we can make it clearer and better. You can either create an issue in the github repo or contact us at jackie_kxz@outlook.com.