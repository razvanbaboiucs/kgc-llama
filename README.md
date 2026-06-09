# Resource-Efficient Fine-Tuning of LLaMA 3.1 for Triple Classification in Knowledge Graph Completion

Code and data for the dissertation thesis of the same name.

## Overview

Knowledge graphs store facts as **triples** `(head, relation, tail)`. **Triple classification** —
deciding whether a triple is **True** or **False** — is a core sub-task of Knowledge Graph
Completion (KGC).

This project reframes it as an instruction-following task and fine-tunes **LLaMA 3.1 (8B)** to
answer it. Fine-tuning is parameter-efficient (4-bit quantization + LoRA via
[Unsloth](https://github.com/unslothai/unsloth)) so it runs on a single Colab GPU. Each triple
becomes an Alpaca-style prompt:

```
Instruction: Is the following knowledge graph triplet True or False?
Input:       Antoine Brutus Menier religion Roman Catholic Church
Response:    True
```

The model is evaluated by exact-match accuracy of its `True` / `False` answer on the test split.

## Datasets

Three benchmarks, one folder each under [`data/`](data/), taken and adapted from
[yao8839836/kg-bert](https://github.com/yao8839836/kg-bert): **FB13** (Freebase facts), **WN11**
(WordNet relations), and **umls** (biomedical concepts).

Each folder contains:

- **`train.tsv` / `dev.tsv` / `test.tsv`** — the triple splits. `train.tsv` is positives only
  (`head <TAB> relation <TAB> tail`); `dev`/`test` add a label column `1` (true) or `-1` (false).
- **`entity2text*.txt` / `relation2text.txt`** — maps from IDs to readable surface forms used to
  verbalize triples (`entity2text_capital.txt` is the FB13 variant used for prompts).
- **`*_instructions_llama_new.json`** — the generated `{instruction, input, output}` data the
  notebook loads, produced by [`prepare-data.py`](prepare-data.py).

> FB13 also ships down-sampled `*-minimal.tsv` splits, so its instruction files are named
> `*_minimal_instructions_llama_new.json`; WN11/umls use the full splits (no `minimal`). Paths and
> the `entity2text*` variant must be adjusted per dataset.

## Repository layout

```
prepare-data.py        # Triples (.tsv) -> Alpaca instruction JSON
model-training.ipynb   # Fine-tune + evaluate LLaMA 3.1 with Unsloth (Colab)
utils/minimize-data.py # Randomly down-sample a .tsv split
utils/plot-loss.py     # Plot training vs. validation loss
data/                  # FB13, WN11, umls
diagrams/              # Saved loss curves and result figures
```

## How to run

**1. Prepare the instruction data** (configured for FB13; edit the `datasets` list / paths at the
top of the script for WN11 or umls):

```bash
python prepare-data.py
```

For train/dev it emits each positive triple plus one corrupted negative; for test it uses the
existing `1`/`-1` label. Optionally down-sample first:

```bash
python utils/minimize-data.py data/FB13/train.tsv data/FB13/train-minimal.tsv --keep 0.2
```

**2. Fine-tune and evaluate** — open [`model-training.ipynb`](model-training.ipynb) in **Google
Colab** (GPU runtime). It mounts Drive (expecting the JSONs under `drive/MyDrive/kgc-llama/<DATASET>/`),
loads `unsloth/Meta-Llama-3.1-8B` in 4-bit with LoRA, trains with TRL's `SFTTrainer` (1000 steps),
saves the adapters, and reports test accuracy. Switch datasets via the `DATA_PATH` /
`EVAL_DATA_PATH` / `TEST_DATA_PATH` constants.

**3. Inspect results** — plot the logged loss curves:

```bash
python utils/plot-loss.py losses.txt   # "Epoch <n>:<train>:<val>" lines; omit arg for example data
```

Saved figures are in [`diagrams/`](diagrams/).

## Requirements

CUDA GPU (Colab). The notebook installs `unsloth`, `torch`, `transformers`, `trl`, `datasets`; the
local scripts need `pandas` and `matplotlib`.
