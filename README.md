# Probing Tasks for Noised Back-Translation

This repository contains the code necessary to reproduce the experiments and results in the bachelor's thesis Probing "Tasks for Noised Back-Translation" by Nicolas Spring.



## General Remarks

- [noisy-text](https://github.com/valentinmace/noisy-text) by Valentin Macé was used to add noise to back-translations.
- [fairseq-states](https://github.com/nicolasspring/fairseq-states/), a fork of [fairseq](https://github.com/pytorch/fairseq) with added state saving functionality, was used to extract model states.
- The Perl scripts in [diversity](https://github.com/emjotde/diversity) by Marcin Junczys-Dowmunt were used to calculate values for lexical diversity.



## Remarks Regarding the Scripts

- [conda](https://docs.conda.io/en/latest/) was used to create a Python 3.6 environment.
- [slurm](https://slurm.schedmd.com/documentation.html) was used to submit jobs. You may need to tweak the scripts to reproduce results on your machine.



## Reproducing the Results

```bash
git clone https://github.com/nicolasspring/bt-probing-tasks/
cd ./bt-probing-tasks/
```

After cloning the repository to your machine, there are the following steps:



### 1. Creating a Virtual Environment

Create an environment with Python 3.6 using `conda`:

```bash
bash scripts/create_virtualenv.sh
```

**After running this script, please activate the environment. All further steps assume that the environment has been activated.**



### 2. Installing Software

```bash
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



### 3. Downloading and Preprocessing Data

Download the parallel and monolingual target data for model training and back-translation:

```bash
bash scripts/data/download_data.sh
```

Preprocess and binarize the previously downloaded data:

```bash
bash scripts/data/preprocess_data.sh
```



### 4. Training a Reverse Model for Back-Translation

Train a reverse (de-en) model for back-translating the monolingual target data:

```bash
bash scripts/training/train_reverse_model.sh
```

Evaluate the model to make sure it is well trained:

```bash
bash scripts/evaluation/evaluate_reverse_model.sh
```



### 5. Obtaining Back-Translations

#### Beam Search Back-Translations

Perform back-translation over the monolingual target data with beam size 5:

```bash
bash scripts/bt/generate_beam_bt.sh
```

Combine the shards and extract back-translations from the fairseq output:

```bash
bash scripts/bt/extract_beam_bt.sh
```

#### Noised Back-Translations

Add noise to the back-translations obtained with beam search to create a noised back-translation dataset:

```bash
bash scripts/bt/generate_noised_bt.sh
```

#### Tagged Back-Translations

Add `<BT>` tags to the back-translations obtained with beam search to create a tagged back-translation dataset.

```bash
bash scripts/bt/generate_tagged_bt.sh
```

#### Binarizing the Back-Translations

Binarize all three back-translation datasets:

```bash
bash scripts/bt/binarize_all_bt.sh
```

#### Creating Combined Datasets

Use symlinks to create the three different combined datasets for model training:  `wmt18_en_de_para_plus_beam`, `wmt18_en_de_para_plus_noised` and `wmt18_en_de_para_plus_tagged`.

```bash
bash scripts/bt/combine_datasets.sh
```



### 6. Training Models

#### Training the beamBT Model

Train a model with parallel data and beam back-translation:

```bash
bash scripts/training/train_beam_model.sh
```

Evaluate the model to make sure it is well trained:

```bash
bash scripts/evaluation/evaluate_beam_model.sh
```

#### Training the noisedBT Model

Train a model with parallel data and noised back-translation:

```bash
bash scripts/training/train_noised_model.sh
```

Evaluate the model to make sure it is well trained:

```bash
bash scripts/evaluation/evaluate_noised_model.sh
```

#### Training the taggedBT Model

Train a model with parallel data and tagged back-translation:

```bash
bash scripts/training/train_tagged_model.sh
```

Evaluate the model to make sure it is well trained:

```bash
bash scripts/evaluation/evaluate_tagged_model.sh
```



### 7. Extracting Model States for the Probing Task

#### Creating Train and Test Datasets

```bash
bash scripts/probing_tasks/create_probing_task_datasets.sh
```

This script creates training and test sets for the probing tasks from the parallel data. It back-translates the data to create parallel (original and back-translated) datasets from which model states can be extracted in the next step.

#### Extracting Model States

```bash
bash scripts/probing_tasks/extract_model_states.sh
```

This script will use the datasets previously created and save model states for two experiments: **genuine source text vs. back-translation** and **genuine source text vs. noised back-translation**.

#### Checking the Dimensions of the Extracted States

Sanity checks: This script checks if all states have the correct dimensions. It compares the C axis (`encoder_embed_dim`) to the model configuration and checks if no batches are larger than the batch size specified in the training commands. In the probing task step, states will be padded to the highest number of time steps in the training set.

Training set for **genuine source text vs. back-translation** :

```bash
python scripts/checks/check_state_shapes.py \
	model_states/en_de_parallel_plus_bt_beam/bitext/ \
	model_states/en_de_parallel_plus_bt_beam/beam/
```

Training set for **genuine source text vs. noised back-translation** :

```bash
python scripts/checks/check_state_shapes.py \
	model_states/en_de_parallel_plus_bt_noised/bitext/ \
	model_states/en_de_parallel_plus_bt_noised/noised/
```

In the script `scripts/probing_tasks/run_probing_tasks.sh`, please adjust the flag `--max-len` to the maximum sequence length. The default value is 246, but depending on the random split, your sequence length may be different.



### 8. Training Models (Probing Task)

Combine the model states and use them as features for the probing tasks:

```bash
bash scripts/probing_tasks/run_probing_tasks.sh
```

This script creates a directory each in `./probing_tasks/` for the two experiments. The directories contain CSV files with the results and the pickled classifiers.



### 9. Analyzing Generation With and Without the `<BT>` Tag

To perform qualitative analysis on the taggedBT model, generate back-translations with and without the `<BT>` tag:

```bash
bash scripts/qualitative_analysis/generate_qualitative_analysis_data.sh
```

This script translates the test sets and creates raw and postprocessed output files.

#### BLEU Scores

To calculate BLEU scores on newstest2017 and newstest2019:

```bash
# newstest2014
bash scripts/evaluation/translate_newstest2014_taggedBT.sh
# newstest2017
bash scripts/evaluation/translate_newstest2017_taggedBT.sh
# newstest2019
bash scripts/evaluation/translate_newstest2019_taggedBT.sh
```

For newstest2019, scores are calculated on the WMT reference translation as well as the additional references from [Freitag et al. (2020)](https://arxiv.org/abs/2004.06063).

#### Lexical Diversity

To calculate copy-aware lexical diversity for newstest2014, newstest2017 and newstest2019:

```bash
bash scripts/qualitative_analysis/calculate_lexical_diversity.sh
```

Note: In the script, `no_counts` occupies the argument slot used for an English language word count list. When working with German, this argument does no influence the script.

#### Alignment Distances

To train a `fast_align` model on the parallel data and align newstest2014, newstest2017 and newstest2019:

```bash
bash scripts/qualitative_analysis/align_data.sh
```

To calculate the normalized distance between corresponding words for the translations:

```bash
# newstest2014
# translation generated with tag
python scripts/qualitative_analysis/measure_parallelism.py \
    -i qualitative_analysis/alignment/output/wmt14.aligned.en-de_tag

# translation generated without the tag
python scripts/qualitative_analysis/measure_parallelism.py \
    -i qualitative_analysis/alignment/output/wmt14.aligned.en-de_no_tag


# newstest2017
# translation generated with tag
python scripts/qualitative_analysis/measure_parallelism.py \
	-i qualitative_analysis/alignment/output/wmt17.aligned.en-de_tag

# translation generated without the tag
python scripts/qualitative_analysis/measure_parallelism.py \
	-i qualitative_analysis/alignment/output/wmt17.aligned.en-de_no_tag


# newstest2019
# translation generated with tag
python scripts/qualitative_analysis/measure_parallelism.py \
	-i qualitative_analysis/alignment/output/wmt19.aligned.en-de_tag

# translation generated without the tag
python scripts/qualitative_analysis/measure_parallelism.py \
	-i qualitative_analysis/alignment/output/wmt19.aligned.en-de_no_tag
```

