# Transliteration Model – with Attention**  
*(DA6401 – Assignment 3, Part B)*

This repository contains an **encoder-decoder** model with a **attention** mechanism for character-level transliteration (e.g. English → Devanagari).  
The implementation is written in **PyTorch Lightning**, supports **LSTM / GRU / vanilla RNN** back-ends, DATASET USED DAKSHINA DATASET.

All experiments were executed on **Kaggle Notebooks** using GPU.

* **GitHub (code)** <https://github.com/Pavitra-khare/DA6401_ASS3_withAtten>   
* **Weights & Biases report** <https://wandb.ai/your-entity/DA6401_Ass3_Attention/reports>
* ***check sweeps here*** [Sweeps on kaggle](https://www.kaggle.com/code/pavitrakharecs24m031/attentionass3?scriptVersionId=240636444)

---

## Objective

* Map variable-length Latin words to their target-script transliterations.
* Explore the impact of **attention**, **bidirectionality**, depth, and dropout.
* Provide clean data-ingest utilities and  visualise attention weight heat-maps.
* Log every run (hyper-parameters, metrics, predictions) to **Weights & Biases**.

---

## Model Architecture

| Component | Details |
|-----------|---------|
| **Encoder** | Embedding → *n*-layer (**uni/bi**) RNN/GRU/LSTM |
| **Attention** | Dot-product between last decoder hidden state and every encoder timestep |
| **Decoder** | Embedding → Attention-context concat → *m*-layer RNN/GRU/LSTM → Linear **vocab** |
| **Loss** | `CrossEntropyLoss` |
| **Accuracy** | Word-level exact match |

> The concrete classes live in **`classOfEncoderWithAttn`**, **`classOfDecoderWithAttn`**, and the high-level Lightning module **`withAttentionSeqToSeq`** inside `modularFinalAttention.py`.

---

##  Key Functions / Classes

| Name | Type | Purpose |
|------|------|---------|
| `get_config()` | function | Parse CLI flags, log-in to WandB, pick device |
| `char2indices`, `word2indfunc` | function | Character → index tensor (adds BOS/EOS, pads/truncates) |
| `getDataLoaders()` | function | Build `DataLoader`s for **train / val / test** CSV files |
| `classOfEncoderWithAttn` | class | Stacked, (bi-)directional RNN encoder |
| `classOfDecoderWithAttn` | class | Decoder with **dot-product attention** step |
| `withAttentionSeqToSeq` | class | LightningModule: training loop, accuracy, logging |
| `plotHeatMap()` | function | 3 × 3 grid of attention matrices (matplotlib + seaborn) |
| `save_outputs_to_csv()` | function | Append *input / gold / prediction* triplets to `Output.csv` |

---

## Command-Line Flags

The script uses `argparse`; the most relevant options are:

| Flag | Default | Description |
|------|---------|-------------|
| `--train`, `--val`, `--test` | *CSV paths* | Dakshina dataset splits |
| `--hidden_layer_size` | `256` | RNN hidden units |
| `--embedding_size` | `128` | Character embedding size |
| `--encoder_layers` / `--decoder_layers` | `3 / 3` | Depth of the stacks |
| `--cell_type` | `LSTM` | `LSTM`, `GRU`, or `RNN` |
| `--bidirectional` | `True` | Bidirectional encoder |
| `--dropout` | `0.5` | Dropout inside recurrent layers |
| `--epochs` | `20` | Training epochs |
| `--attention` | **`True`** | (Kept for parity – always on in this script) |
| `--wandb_project`    | `"DA6401_ASS3_ATTENTION"`                                  | Project name on wandb                    |
| `--wandb_entity`     | `"3628-pavitrakhare-indian-institute-of-technology-madras"` | Your wandb user or team               |
| `--key`              | `"<wandb-api-key>"`                                      | wandb authentication key                 |

---

## Evaluation

* **Loss** – token cross-entropy (PAD/BOS excluded)  
* **Accuracy** – *word-level* : a prediction counts only if **all** inner tokens match the gold string.  
* Results plus attention plots are logged to **WandB**.

---

## How to Run

### 1 · Install Dependencies

```bash
# Core Deep-Learning stack
pip install torch torchvision torchaudio
pip install pytorch-lightning

# Utilities & visualisation
pip install pandas numpy matplotlib seaborn
pip install wandb

```
### 2 . Run the program
```bash
python train.py --train <path_to_train.csv> --val <path_to_val.csv> --test <path_to_test.csv>
```
