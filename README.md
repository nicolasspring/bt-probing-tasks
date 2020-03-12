# Probing Tasks for Noised Back-Translation

This repository contains the code necessary to reproduce the experiments and results in the bachelor's thesis Probing "Tasks for Noised Back-Translation" by Nicolas Spring. The thesis is concerned with the theory proposed by [Caswell et al. (2019)](https://arxiv.org/pdf/1906.06442.pdf) that noise in back-translation (BT) serves as an implicit label. The code in this repository was used to train the NMT models in [Edunov et al. (2018)](https://arxiv.org/pdf/1808.09381.pdf), extract model states and train [sklearn](https://scikit-learn.org/stable/) classifiers on them.



## General Remarks

- [noisy-text](https://github.com/valentinmace/noisy-text) by Valentin Macé was used to add noise to back-translations.
- [fairseq-states](https://github.com/nicolasspring/fairseq-states/), a fork of [fairseq](https://github.com/pytorch/fairseq) with added state saving functionality, was used to extract model states.



## Remarks Regarding the Scripts

- [conda](https://docs.conda.io/en/latest/) was used to create a Python 3.6 environment.
- [slurm](https://slurm.schedmd.com/documentation.html) was used to submit jobs. You may need to tweak the scripts to reproduce results on your machine.



## How to Reproduce the Results

```
git clone https://github.com/nicolasspring/bt-probing-tasks/
cd ./bt-probing-tasks/
```

After cloning the repository to your machine, there are the following steps:



### Creating a Virtual Environment

Create an environment with Python 3.6 using `conda`:

```
bash scripts/create_virtualenv.sh
```

**After running this script, please activate the environment. All further steps assume that the environment has been activated.**



### Installing Software

```
bash scripts/install_packages.sh
```

This scripts will install the following packages:

- `torch`
- `fairseq`
- `fastBPE`
- `sacremoses`
- `subword_nmt`
- `numpy`
- `scipy`
- `sklearn`
- `matplotlib`
- `pandas`



### Downloading and Preprocessing Data

Download the parallel and monolingual target data for model training and back-translation:

```
bash scripts/data/download_data.sh
```

Preprocess and binarize the previously downloaded data:

```
bash scripts/data/preprocess_data.sh
```



### Training a Reverse Model for Back-Translation

Train a reverse (de-en) model for back-translating the monolingual target data:

```
bash scripts/training/train_reverse_model.sh
```



### Obtaining Back-Translations

#### Beam Search Back-Translations

Perform back-translation over the monolingual target data with beam size 5:

```
bash scripts/bt/generate_beam_bt.sh
```

Combine the shards and extract back-translations from the fairseq output:

```
bash scripts/bt/extract_beam_bt.sh
```

#### Noised Back-Translations

Add noise to the back-translations obtained with beam search to create a noised back-translation dataset:

```
bash scripts/bt/generate_noised_bt.sh
```

#### Tagged Back-Translations

Add `<BT>` tags to the back-translations obtained with beam search to create a tagged back-translation dataset.

```
bash scripts/bt/generate_tagged_bt.sh
```

#### Binarizing the Back-Translations

Binarize all three back-translation datasets:

```
bash scripts/bt/binarize_all_bt.sh
```

#### Creating Combined Datasets

Use symlinks to create the three different combined datasets for model training:  `wmt18_en_de_para_plus_beam`, `wmt18_en_de_para_plus_noised` and `wmt18_en_de_para_plus_tagged`.

```
bash scripts/bt/combine_datasets.sh
```



### Training Models

#### Training the beamBT Model

Train a model with parallel data and beam back-translation:

```
bash scripts/training/train_beam_model.sh
```

#### Training the noisedBT Model

Train a model with parallel data and noised back-translation:

```
bash scripts/training/train_noised_model.sh
```

#### (Optional) Training the taggedBT Model

Train a model with parallel data and tagged back-translation:

```
bash scripts/training/train_tagged_model.sh
```



### Extracting Model States for the Probing Task

#### Creating Train and Test Datasets

```
bash scripts/probing_tasks/create_probing_task_datasets.sh
```

This script creates training and test sets for the probing tasks from the parallel data. It back-translates the data to create parallel (original and back-translated) datasets from which model states can be extracted in the next step.

#### Extracting Model States

```
bash scripts/probing_tasks/extract_model_states.sh
```

This script will use the datasets previously created and save model states for two experiments: **genuine source text vs. back-translation** and **genuine source text vs. noised back-translation**.
