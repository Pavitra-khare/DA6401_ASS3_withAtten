# ────────────────────────────── third-party libs ─────────────────────────────
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils import data as torch_data
from IPython.display import display, HTML
import pytorch_lightning as pl
import wandb

import numpy as np
import pandas as pd

from matplotlib.font_manager import FontProperties
from itertools import chain
# ──────────────────────────────── standard lib ───────────────────────────────
import argparse
import csv
import os

import seaborn as sns
import matplotlib.pyplot as plt
# ────────────────────────────── house-keeping ────────────────────────────────
torch.cuda.empty_cache()          # free any residual GPU memory







# ──────────────────────────────────────────────────────────────────────────────
# configuration function
# ──────────────────────────────────────────────────────────────────────────────

def get_config():
    """
    Parse command-line flags, log in to Weights-and-Biases, and return both the
    populated `args` namespace and the chosen torch `device`.

    Down-stream code can still rely on:
        args.hidden_layer_size, args.encoder_layers, args.bidirectional …
    """
    # DA6401_ASS3_ATTENTION

    
    flag_spec = [
        (("-p", "--wandb_project"),     dict(default="final Attention")),
        (("-e", "--wandb_entity"),      dict(default="3628-pavitrakhare-indian-institute-of-technology-madras")),
        (("--key",),                   dict(dest="wandb_key",
                                            default="71aebd5eed3e9b3e37a5a3c4658f5433375d97dc")),
        (("-tr", "--train"),           dict(dest="trainFilepath",
                                            default="/kaggle/input/dakshina/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv")),
        (("-va", "--val"),             dict(dest="valFilePath",
                                            default="/kaggle/input/dakshina/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv")),
        (("-te", "--test"),            dict(dest="testFilePath",
                                            default="/kaggle/input/dakshina/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv")),
        (("-hs", "--hidden_layer_size"), dict(type=int, default=512)),
        (("-es", "--embedding_size"),    dict(type=int, default=64)),
        (("-enc", "--encoder_layers"),   dict(type=int, default=3)),
        (("-dec", "--decoder_layers"),   dict(type=int, default=3)),
        (("-bi", "--bidirectional"),     dict(action="store_true", dest="bidirectional",default=True)),
        (("-ct", "--cell_type"),         dict(default="LSTM")),
        (("--attention",),               dict(action="store_true",default=True)),
        (("--epochs",),                  dict(type=int, default=15)),
        (("--dropout",),                 dict(dest="drop_out", type=float, default=0.5)),
        (("-lr", "--learning_rate"),    dict(dest="learning_rate",type=float, default=0.001)),
    ]

    parser = argparse.ArgumentParser()
    for flags, kwargs in flag_spec:
        parser.add_argument(*flags, **kwargs)

    args = parser.parse_args()

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   
    wandb.login(key=args.wandb_key, relogin=True)
    wandb.init(project=args.wandb_project,
               entity=args.wandb_entity,
               config=vars(args))

    return args, device








# ─────────────────────────────────────────────────────────────────────────────
# 1.  HELPER  ▸  CSV / DATA INGEST
# ─────────────────────────────────────────────────────────────────────────────

def read_file_0(filepath):
    #read the 0th col of input file
    def process_row(text_row):
        return list(text_row)

    def collect_characters(reader_obj):
        all_chars = []
        for entry in reader_obj:
            all_chars += process_row(entry[0])
        return all_chars

    with open(filepath, mode='r') as file_handle:
        csv_reader = csv.reader(file_handle,delimiter='\t')
        return collect_characters(csv_reader)
    



def read_file_1(trainFilepath):
    #read the 1st col of input file
    def extract_chars(text):
        return list(text)

    def accumulate(reader_obj):
        buffer = []
        for entry in reader_obj:
            buffer += extract_chars(entry[1])
        return buffer

    with open(trainFilepath, 'r') as file_stream:
        csv_rows = csv.reader(file_stream,delimiter='\t')
        return accumulate(csv_rows)
    
    


def _char_index_from_csv(csv_path: str, column: int) -> dict[str, int]:
    """
 
    CSV, add the special delimiter '|', and return a 1-based index dictionary.
    """
    with open(csv_path, newline='') as fh:
        reader = csv.reader(fh,delimiter='\t')
        # flatten characters from all rows in that column
        charset = set(chain.from_iterable(row[column] for row in reader))

    charset.add('|')                           # guarantee delimiter presence
    return {ch: idx + 1 for idx, ch in enumerate(charset)}   # 1-based IDs




def getCharToIndex(args):
    """Character → index map for the *Latin* (source) side – column 0."""
    return _char_index_from_csv(args.trainFilepath, 1)



def getCharToIndLang(args):
    """Character → index map for the *target language* side – column 1."""
    return _char_index_from_csv(args.trainFilepath, 0)








# ─────────────────────────────────────────────────────────────────────────────
# 2.  HELPER  ▸  TEXT → TENSOR CONVERSION
# ─────────────────────────────────────────────────────────────────────────────
def char2indices(word, dictionary):
    # Convert each character in the word to its index using the dictionary.
    # Filters out characters not present in the dictionary.
    def safe_lookup(char):
        return dictionary[char] if char in dictionary else -1

    mapped = [safe_lookup(ch) for ch in word]
    filtered = list(filter(lambda idx: idx >= 0, mapped))
    return filtered


def changeLengthOfSeq(indices, maximumLength):
    # Adjusts a sequence's length by trimming or padding with zeros.
    # Ensures consistent input length for neural models.
    current_length = len(indices)

    def trim(seq, length):
        return seq[:length]

    def pad(seq, total_length):
        padding_needed = total_length - len(seq)
        return seq + [0] * padding_needed

    if current_length > maximumLength:
        return trim(indices, maximumLength)
    elif current_length < maximumLength:
        return pad(indices, maximumLength)
    return indices


def get_keyAttn(index_value, charToIndLang):
    # Finds the character corresponding to a given index.
    # Returns an empty string if index is not found.
    for character, idx in charToIndLang.items():
        if idx == index_value:
            return character
    return ""


def ind2Ten(device, indices, dictionary):
    # Adds BOS and EOS tokens to the index list and converts to tensor.
    # Uses '|' as the delimiter token.
    token = dictionary.get('|', 0)

    def add_delimiters(seq, token_id):
        return [token_id] + seq + [token_id]

    sequence = add_delimiters(indices, token)
    tensor_obj = torch.tensor(sequence, device=device)
    return tensor_obj


def word2indfunc(device, word, maximumLength, dict):
    # Converts word to a fixed-length tensor of character indices.
    # Includes BOS/EOS and pads/trims as needed.
    char_ids = char2indices(word, dict)

    def standardize_length(seq, target_len):
        return changeLengthOfSeq(seq, target_len)

    resized = standardize_length(char_ids, maximumLength)
    tensor_out = ind2Ten(device, resized, dict)
    return tensor_out









# ─────────────────────────────────────────────────────────────────────────────
# 3.  HELPER  ▸  DATASET-SPECIFIC UTILITIES
# ─────────────────────────────────────────────────────────────────────────────


def getMaxLenEng(args):
    # Returns the maximum length of any English word in the training file.
    # Scans column 0 of the CSV to compute max word length.
    with open(args.trainFilepath, 'r') as csv_file:
        reader = csv.reader(csv_file,delimiter='\t')
        max_length = 0

        def update_max(current_max, candidate):
            return candidate if candidate > current_max else current_max

        for record in reader:
            length = len(record[1])
            max_length = update_max(max_length, length)

        maxLenEng = max_length
    return maxLenEng


def getMaxLenDev(args):
    # Returns the maximum length of any Devanagari word in the training file.
    # Scans column 1 of the CSV to find the longest word.
    with open(args.trainFilepath, 'r') as csv_stream:
        reader_obj = csv.reader(csv_stream,delimiter='\t')
        longest = 0

        def get_larger(a, b):
            return b if b > a else a

        for line in reader_obj:
            current_length = len(line[0])
            longest = get_larger(longest, current_length)

        maxLenDev = longest
    return maxLenDev


def keyForInput(val, char_to_idx_latin):
    # Given an index, returns the corresponding Latin character.
    # Used to decode model predictions into readable strings.
    for k, v in char_to_idx_latin.items():
        if val == v:
            return k
    return ""


def keyForVal(val, charToIndLang):
    # Given an index, returns the corresponding Devanagari character.
    # Used for reconstructing predicted target words.
    for k, v in charToIndLang.items():
        if val == v:
            return k
    return ""


def generate_indices(device, row, char_to_idx_latin, charToIndLang, maxLenEng, maxLenDev):
    # Converts a pair of (eng, dev) words into padded tensor form using vocab maps.
    # Applies fixed-length encoding using BOS/EOS and returns input/output tensors.
    src_text = row[1]
    tgt_text = row[0]

    def build_indexed_tensor(text, max_len, char_map):
        return word2indfunc(device, text, max_len, char_map)

    source_tensor = build_indexed_tensor(src_text, maxLenEng, char_to_idx_latin)
    target_tensor = build_indexed_tensor(tgt_text, maxLenDev, charToIndLang)

    return source_tensor, target_tensor








# ─────────────────────────────────────────────────────────────────────────────
# 4.  HELPER  ▸  DATALOADER FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def getDataLoaders(device, args, char_to_idx_latin, charToIndLang, maxLenEng, maxLenDev):
    # Loads and processes train, val, and test CSVs into tensor pairs.
    # Returns three DataLoader objects for model training, validation, and testing.
    
    batch_size = 32
    train_shuffle = True
    eval_shuffle = False

    # ── Validation dataset ──
    pairs_v = []
    with open(args.valFilePath, 'r') as val_file:
        csv_reader = csv.reader(val_file,delimiter='\t')

        def process_and_append(pairs, record):
            eng_tensor, hin_tensor = generate_indices(device, record, char_to_idx_latin, charToIndLang, maxLenEng, maxLenDev)
            pairs.append([eng_tensor, hin_tensor])

        for entry in csv_reader:
            process_and_append(pairs_v, entry)

    # ── Test dataset ──
    pairs_t = []
    with open(args.testFilePath, 'r') as test_file:
        test_reader = csv.reader(test_file,delimiter='\t')

        def append_indexed_pair(container, data_row):
            src_tensor, tgt_tensor = generate_indices(device, data_row, char_to_idx_latin, charToIndLang, maxLenEng, maxLenDev)
            container.append([src_tensor, tgt_tensor])

        for record in test_reader:
            append_indexed_pair(pairs_t, record)

    # ── Train dataset ──
    pairs = []
    with open(args.trainFilepath, 'r') as train_file:
        train_reader = csv.reader(train_file,delimiter='\t')

        def process_row_and_store(storage, row_data):
            input_tensor, target_tensor = generate_indices(device, row_data, char_to_idx_latin, charToIndLang, maxLenEng, maxLenDev)
            storage.append([input_tensor, target_tensor])

        for entry in train_reader:
            process_row_and_store(pairs, entry)

    # ── Dataloader creation ──
    def create_loader(dataset, size, shuffle_flag):
        return torch.utils.data.DataLoader(dataset, batch_size=size, shuffle=shuffle_flag)

    dataloaderTrain = create_loader(pairs, batch_size, train_shuffle)
    dataloaderVal = create_loader(pairs_v, batch_size, eval_shuffle)
    dataloaderTest = create_loader(pairs_t, batch_size, eval_shuffle)

    return dataloaderTrain, dataloaderVal, dataloaderTest









# ─────────────────────────────────────────────────────────────────────────────
# 5.  HELPER  ▸  LOGGING / I-O
# ─────────────────────────────────────────────────────────────────────────────


def save_outputs_to_csv(inputs, targets, predictions):
    


    records = {
        'input': inputs,
        'target': targets,
        'predicted': predictions
    }
    output_file = 'Output.csv'

    dataframe = pd.DataFrame(records)
    already_present = os.path.isfile(output_file)
    dataframe.to_csv(output_file, mode='a', index=False, header=not already_present)


def log_predictions_to_wandb(inputs, targets, preds):
    # Logs predictions to Weights & Biases with ✅/❌ labels for correctness.
    # Creates a wandb.Table object for visualization in the dashboard.

    prediction_table = wandb.Table(columns=["Input", "Target", "Prediction Result"])

    for src, gold, guess in zip(inputs, targets, preds):
        result = "✅" if gold == guess else "❌"
        labeled_pred = f"{guess} {result}"
        prediction_table.add_data(src, gold, labeled_pred)

    wandb.log({"Predictions Overview": prediction_table})









# ─────────────────────────────────────────────────────────────────────────────
# 6.  HELPER  ▸  Tensors related functions
# ─────────────────────────────────────────────────────────────────────────────

def mapTensor2seq(sequence):
    # Joins space-separated tokens into a continuous string.
    # Returns the joined string and its total character count.

    tokens = sequence.split()

    def concatenate(tokens):
        return ''.join(tokens)

    def compute_total_length(tokens):
        return sum(len(token) for token in tokens)

    combined = concatenate(tokens)
    total_len = compute_total_length(tokens)

    return combined, total_len


def grpTheGeneratedSeq(path):
    # Segments a flattened string into fixed-size chunks for processing.
    # Chunk size is determined dynamically based on input length.

    combined_seq, total_chars = mapTensor2seq(path)

    def determine_chunk_size(length, divisor=4):
        return max(1, length // divisor)

    segment_length = determine_chunk_size(total_chars)
    partitioned = assemble_tensor(combined_seq, segment_length)

    return partitioned


def assemble_tensor(final_tensor, partition_size=1):
    # Breaks the input string into segments of specified size.
    # Returns a list of these equally sized chunks.

    def safe_partition_size(size):
        return max(1, size)

    chunk_size = safe_partition_size(partition_size)
    segments = [final_tensor[idx:idx + chunk_size] for idx in range(0, len(final_tensor), chunk_size)]

    return segments






# ─────────────────────────────────────────────────────────────────────────────
# 6.  HELPER  ▸  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────


def plotHeatMap(input_words, output_words, attentionWeights):
    # Plots attention weights between input and output tokens.
    # Displays up to 9 attention maps in a 3×3 grid layout.

    dev_font = FontProperties(fname="/kaggle/input/devnagri/Nirmala.ttf")

    fig, axarr = plt.subplots(3, 3, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    n_plots = min(len(input_words), len(output_words), len(attentionWeights))

    for idx, ax in enumerate(axarr.ravel()):
        if idx >= n_plots:  # Hide unused subplots
            ax.axis("off")
            continue

        # Extract attention matrix for this (input, output) pair
        attn = attentionWeights[idx].cpu().detach().numpy()
        attn = attn[1 : len(input_words[idx]) + 1, : len(output_words[idx])]

        # Draw heatmap of attention weights
        sns.heatmap(
            attn,
            ax=ax,
            cmap="viridis",  # Color scale from purple to yellow
            cbar=False,
            square=True,
            vmin=0,
            vmax=attn.max(),
        )

        # Draw dashed grid lines on heatmap
        ax.set_xticks(np.arange(-0.5, attn.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, attn.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="--", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)

        # Set x and y axis labels using token sequences
        ax.xaxis.tick_top()
        ax.set_xticks(range(len(output_words[idx])))
        ax.set_xticklabels(
            reversed(output_words[idx]),
            rotation=45,
            ha="right",
            fontproperties=dev_font,
            fontsize=9,
        )

        ax.set_yticks(range(len(input_words[idx])))
        ax.set_yticklabels(
            reversed(input_words[idx]),
            rotation="vertical",
            fontproperties=dev_font,
            fontsize=9,
        )

        ax.set_title(f"Attention {idx + 1}", fontsize=12)

    # Log to wandb and display plot
    wandb.log({"Question5_HeatMap": wandb.Image(fig)})
    plt.show()
    plt.close(fig)


def score2color(value):
    color_palette = [
        '#cce5ff', '#b3d7f2', '#99c9e6', '#80bbd9', '#66adcd',
        '#4d9fc0', '#3391b3', '#1a83a6', '#007599', '#fbe9e9',
        '#f6cccc', '#f2aeae', '#ed9191', '#e87575', '#e35858',
        '#df3b3b', '#da1f1f', '#d50303', '#b80000', '#9c0000'
    ]
    index = min(len(color_palette) - 1, int((value * 100) / 5))
    return color_palette[index]

# -------------------- HTML span for colored characters --------------------
def colotForChar(char, hex_color):
    if char == ' ':
        return f"<span style='padding: 2px 4px;'>{char}</span>"
    return f"<span style='color: {hex_color}; padding: 2px 4px;'>{char}</span>"

# -------------------- IPython display function --------------------
def render_colored_text(char_color_pairs):
    styled_line = ''.join(colotForChar(char, color) for char, color in char_color_pairs)
    display(HTML(styled_line))

# -------------------- Main visualisation logic --------------------
def visualizationOfQ6(input_sequence, output_sequence, attention_matrix):
    for out_idx, out_char in enumerate(output_sequence):
        print(f"\nFocus on Output Character: '{out_char}'\n")
        row_attention = attention_matrix[out_idx]
        colored_input = [(input_sequence[in_idx], score2color(score))
                         for in_idx, score in enumerate(row_attention)]
        render_colored_text(colored_input)


# -------------------- HTML export version --------------------
def generate_connectivity_html(input_list, output_list, attention_list, output_path=os.getcwd()):
    html = """
    <html>
    <head>
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
            font-family: Arial, sans-serif;
        }
        th, td {
            border: 1px solid #333;
            padding: 10px;
            text-align: center;
            vertical-align: top;
        }
        .char-box {
            display: inline-block;
            padding: 2px 6px;
            margin: 1px;
            font-size: 16px;
            font-family: monospace;
        }
        .native-row {
            margin-bottom: 10px;
        }
        .native-char {
            font-weight: bold;
            display: block;
            margin-bottom: 4px;
        }
    </style>
    </head>
    <body>
    <table>
        <tr>
            <th><i>ID</i></th>
            <th><b>Predicted Latin word for corresponding<br>Native word</b></th>
            <th><b>Native word’s character’s weightage</b></th>
        </tr>
    """

    for idx, (input_seq, output_seq, attn_matrix) in enumerate(zip(input_list, output_list, attention_list)):
        prediction_str = "".join(input_seq)
        native_str = "".join(output_seq)

        attn_html = ""
        for out_char, row_attn in zip(output_seq, attn_matrix):
            row_html = f"<div class='native-row'><span class='native-char'>{out_char}</span><div>"
            for in_char, weight in zip(input_seq, row_attn):
                color = score2color(weight)
                row_html += f"<span class='char-box' style='background-color:{color}'>{in_char}</span>"
            row_html += "</div></div>"
            attn_html += row_html

        html += f"""
        <tr>
            <td><b>{200+idx}</b></td>
            <td>{native_str} &nbsp; &lt;=&nbsp; <code>{prediction_str}</code></td>
            <td>{attn_html}</td>
        </tr>
        """

    html += """
    </table>
    </body>
    </html>
    """

    with open(os.path.join(output_path, "Q6.html"), "w", encoding="utf-8") as f:
        f.write(html)




# ─────────────────────────────────────────────────────────────────────────────
# 7.  Encoder, Decoder class with attention
# ─────────────────────────────────────────────────────────────────────────────




class classOfEncoderWithAttn(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, rnn_type, dropout, num_layers, is_bidirectional):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.is_bidirectional = is_bidirectional

        # Embedding layer to map token indices to dense vectors
        self.embedding = nn.Embedding(input_dim, embed_dim)

        # Map string rnn_type to actual PyTorch RNN module
        rnn_map = {
            "GRU": nn.GRU,
            "LSTM": nn.LSTM,
            "RNN": nn.RNN
        }

        # Instantiate the chosen RNN variant with the specified config
        self.rnn = rnn_map[rnn_type](embed_dim, hidden_dim, dropout=dropout, num_layers=num_layers, bidirectional=is_bidirectional)

    def forward(self, x):
        # Embed input sequence and pass it through the RNN
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        return output, hidden







class classOfDecoderWithAttn(nn.Module):
    def __init__(self, output_dim, hidden_dim, embed_dim, rnn_type, dropout, num_layers, is_bidirectional, max_seq_len):
        super().__init__()
        self.max_seq_len = max_seq_len + 2
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.is_bidirectional = is_bidirectional
        self.rnn_type = rnn_type

        # Embedding layer for output vocabulary
        self.embedding = nn.Embedding(output_dim, embed_dim)

        # Linear layer to compute attention scores from (hidden + embedded)
        self.attn = nn.Linear(hidden_dim + embed_dim, self.max_seq_len)

        # Layer to combine context vector and current embedding
        concat_dim = hidden_dim * (2 if is_bidirectional else 1) + embed_dim
        self.attn_combine = nn.Linear(concat_dim, hidden_dim)

        # Initialize RNN cell based on type (GRU/LSTM/RNN)
        rnn_cls = {"GRU": nn.GRU, "LSTM": nn.LSTM, "RNN": nn.RNN}[rnn_type]
        self.rnn = rnn_cls(hidden_dim, hidden_dim, dropout=dropout, num_layers=num_layers, bidirectional=is_bidirectional)

        # Output projection to vocab size
        output_proj_dim = hidden_dim * (2 if is_bidirectional else 1)
        self.output_layer = nn.Linear(output_proj_dim, output_dim)

    def forward(self, token: torch.Tensor, state, enc_out: torch.Tensor):
            """
            Perform a single decoding step with attention mechanism.
            """

            # Step 1: Embed the input token
            emb = self.embedding(token).unsqueeze(0)  # (1, batch, embed_dim)

            # Step 2: Extract last hidden state from encoder output
            h_prev = state[0][-1] if self.rnn_type == "LSTM" else state[-1]

            # Step 3: Compute attention weights from embedded + hidden
            score_inp = torch.cat([emb.squeeze(0), h_prev], dim=1)
            attn_logits = self.attn(score_inp)
            attn_vec = F.softmax(attn_logits, dim=1)

            # Step 4: Use attention vector to get context vector from encoder output
            ctx = torch.bmm(attn_vec.unsqueeze(1), enc_out.permute(1, 0, 2)).squeeze(1)

            # Step 5: Combine context and embedding to form RNN input
            rnn_inp = torch.cat([emb.squeeze(0), ctx], dim=1)
            rnn_inp = F.relu(self.attn_combine(rnn_inp)).unsqueeze(0)

            # Step 6: Run through the decoder RNN
            output, next_state = self.rnn(rnn_inp, state)

            # Step 7: Final projection to output vocabulary space
            logits = self.output_layer(output.squeeze(0))

            return logits, next_state, attn_vec










# ─────────────────────────────────────────────────────────────────────────────
# 8.  High-level model (Seq2Seq / Seq2SeqAttn)
# ─────────────────────────────────────────────────────────────────────────────


class withAttentionSeqToSeq(pl.LightningModule):
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        embed_dim: int,
        rnn_type: str,
        dropout: float,
        enc_layers: int,
        dec_layers: int,
        is_bidir: bool,
        lr: float,
        max_src_len: int,
        charToIndLang: dict,
        char_to_idx_latin: dict,
    ):
        super().__init__()

        # hyper-params / look-ups

        # modules
        self.encoder = classOfEncoderWithAttn(
            input_dim, hidden_dim, embed_dim,
            rnn_type, dropout, enc_layers, is_bidir
        )
        self.dir_mult   = 2 if is_bidir else 1
        self.decoder = classOfDecoderWithAttn(
            output_dim, hidden_dim, embed_dim,
            rnn_type, dropout, dec_layers, is_bidir, max_src_len
        )

        self.enc_layers = enc_layers
        # metric buffers
        self.tr_loss,  self.tr_acc  = [], []
        self.rnn_type   = rnn_type
        self.val_loss, self.val_acc = [], []
        self.char2hin   = charToIndLang          # target-side vocab
        self.test_loss, self.test_acc = [], []
        self.counter = 0
        self.is_bidir   = is_bidir

        # misc
        self.char2lat   = char_to_idx_latin      # source-side vocab
        self._attn_cache   = []
        self.lr        = lr
        self._heatmap_once = False
        self.dec_layers = dec_layers

    
    def forward(self, src, tgt, teacher_ratio: float = 0.5):
        """
        src : (batch, src_len) long
        tgt : (batch, tgt_len) long
        returns logits (tgt_len, batch, vocab)  and  attn_map
        """
        bsz, tgt_len = tgt.shape
        src = src.t()                                    # (src_len, batch)

        attn_map = torch.zeros(
            tgt_len, bsz, self.decoder.max_seq_len, device=self.device
        )
        vocab = self.decoder.output_layer.out_features
        logits = torch.zeros(tgt_len, bsz, vocab, device=self.device)

        enc_out, hidden = self.encoder(src)

        token = tgt[:, 0]                                # BOS
        for t in range(1, tgt_len):
            step_logits, hidden, attn_map[t] = self.decoder(
                token, hidden, enc_out
            )
            logits[t] = step_logits
            teacher_force = torch.rand(1).item() < teacher_ratio
            token = tgt[:, t] if teacher_force else step_logits.argmax(1)

        return logits, attn_map

    
    def get_model_output(self, src, tgt, teacher_ratio: float = 0.5):
    # Runs forward pass and returns logits in batch-first format
        seq_logits, _ = self(src, tgt, teacher_ratio)
        return seq_logits.permute(1, 0, 2)  # (batch, seq, vocab)

    def prepare_expected_tensor(self, output, target):
        # Converts target sequence into one-hot encoded tensor
        expected = torch.zeros_like(output)
        expected[torch.arange(output.shape[0]), torch.arange(output.shape[1]).unsqueeze(1), target.cpu()] = 1
        return expected

    def prepare_output_expected_tensors(self, seq_logits, tgt):
        # Flattens logits and expected one-hot tensors for loss computation
        expected = self.prepare_expected_tensor(seq_logits, tgt)
        vocab = seq_logits.size(-1)
        return (
            seq_logits[1:].reshape(-1, vocab),
            expected[1:].reshape(-1, vocab),
            tgt[1:].reshape(-1),
        )

    def calculate_loss(self, flat_logits, flat_expected):
        # Calculates cross-entropy loss between logits and targets
        return self.loss_fn(flat_logits.to(self.device), flat_expected.to(self.device))

    def calculate_accuracy(self, batch_logits, tgt):
        # Computes exact word-level accuracy for predicted sequences
        return self.accuracy(batch_logits, tgt)

    def append_metrics(self, loss_val, acc_val):
        # Logs training loss and accuracy for current batch
        self.tr_loss.append(loss_val.detach())
        self.tr_acc.append(torch.tensor(acc_val))

    def training_step(self, batch, batch_idx):
        """
        Run a forward-backward update on one mini-batch and stash the metrics.

        Returns
        -------
        dict
            Must contain the key ``"loss"`` for PyTorch-Lightning.
        """
        
        src, tgt = batch                                    # (B, S_src) , (B, S_tgt)
        seq_logits, _ = self(src, tgt, teacher_ratio=0.5)   # (S_tgt, B, V)
        expected_full = self.prepare_expected_tensor(seq_logits, tgt)   # one-hot

        # ── accuracy (batch-first layout required) ──────
        logits_batch_first = seq_logits.permute(1, 0, 2)            # (B, T, V)
        vocab = seq_logits.shape[-1]

        logits_2d  = seq_logits[1:].contiguous().view(-1, vocab)        # (tokens, V)
        targets_1h = expected_full[1:].contiguous().view(-1, vocab) 
        step_acc = self.accuracy(logits_batch_first, tgt)           # % exact-match

        # ── loss: collapse all time-steps except the initial BOS token ────────────

        step_loss = self.loss_fn(logits_2d, targets_1h)     # scalar

        # ── store for epoch-level averages ──────────
        self.tr_loss.append(step_loss.detach())
        self.tr_acc.append(torch.tensor(step_acc))

        return {"loss": step_loss}






    
    def get_output(self, src, tgt):
    # Performs a forward pass with no teacher forcing and returns both seq and batch-first logits
        seq_logits, _ = self(src, tgt, teacher_ratio=0.0)
        return seq_logits, seq_logits.permute(1, 0, 2)

    def validation_step(self, batch, batch_idx):
        # Computes validation loss and accuracy for a given batch
        src, tgt = batch
        seq_logits, batch_logits = self.get_output(src, tgt)

        flat_logits, expected, _ = self.prepare_output_expected_tensors(seq_logits, tgt)
        val_loss = self.calculate_loss(flat_logits, expected)
        val_acc  = self.calculate_accuracy(batch_logits, tgt)

        self.val_loss.append(val_loss.detach())
        self.val_acc.append(torch.tensor(val_acc))
        return {"loss": val_loss}

    def configure_optimizers(self):
        # Defines the optimizer for training (Adam with specified learning rate)
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def loss_fn(self, output, target):
        # Calculates the average cross-entropy loss between predictions and targets
        return nn.CrossEntropyLoss()(output, target).mean()

    def accuracy(self, batch_logits, tgt):
        # Computes exact-match accuracy by comparing predictions and targets token-wise
        pred = batch_logits.argmax(dim=-1)
        exact = sum(torch.equal(tgt[i, 1:-1], pred[i, 1:-1]) for i in range(tgt.size(0)))
        return 100.0 * exact / tgt.size(0)

    def grid(self, src, output, tgt):
        # Returns the intermediate token sequences: source, target, and predicted outputs
        preds = output.argmax(-1)
        return (
            [src[i, 1:-1]  for i in range(tgt.size(0))],
            [tgt[i, 1:-1]  for i in range(tgt.size(0))],
            [preds[i, 1:-1] for i in range(tgt.size(0))],
        )

    def on_train_epoch_end(self):
        """
        Collate the running lists of loss / accuracy for both splits,
        zero-out the buffers, print a compact summary, and send the
        four scalars to Weights-and-Biases.
        """

        # helper that falls back to 0 when list is empty
        def mean_or_zero(buf):
            return torch.mean(torch.stack(buf)) if buf else torch.tensor(0.0, device=self.device)

        # ── aggregate ---
        stats = {
            "tr_loss": mean_or_zero(self.tr_loss),
            "tr_acc" : mean_or_zero(self.tr_acc),
            "val_loss": mean_or_zero(self.val_loss),
            "val_acc" : mean_or_zero(self.val_acc),
        }

        # ── flush buffers ---
        self.tr_loss.clear();  self.tr_acc.clear()
        self.val_loss.clear(); self.val_acc.clear()

        # ── console print ---
        pretty = {k.replace('_', ' ').title(): round(v.item(), 3) for k, v in stats.items()}
        print(pretty)

        # ── wandb push ---
        wandb.log({
            "Train/Loss"     : stats["tr_loss"],
            "Train/Accuracy" : stats["tr_acc"],
            "Val/Loss"       : stats["val_loss"],
            "Val/Accuracy"   : stats["val_acc"],
        })



   
    def get_out_attention(self, src, tgt):
    # Runs forward pass without teacher forcing, returns logits, attention map, and batch-first logits
        seq_logits, attn = self(src, tgt, teacher_ratio=0.0)
        return seq_logits, attn, seq_logits.permute(1, 0, 2)

    def get_expected(self, output, target):
        # Generates a one-hot encoded tensor of the target for loss computation
        mask = torch.zeros_like(output)
        mask[torch.arange(output.shape[0]), torch.arange(output.shape[1]).unsqueeze(1), target.cpu()] = 1
        return mask

    def test_step(self, batch, batch_idx):
        # Runs inference on test batch, computes loss/accuracy, logs and saves predictions
        grpTheGeneratedSeq('/kaggle/input/dakshina/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv')
        src, tgt = batch
        raw_out, attn_map, acc_out = self.get_out_attention(src, tgt)
        expected = self.get_expected(raw_out, tgt)

        dim = raw_out.shape[-1]
        logits = raw_out[1:].view(-1, dim)
        expected = expected[1:].view(-1, dim)
        labels = tgt[1:].view(-1)

        test_loss = self.loss_fn(logits.to(self.device), expected.to(self.device))
        test_acc = self.accuracy(acc_out, tgt)

        input_tokens, target_tokens, pred_tokens = self.grid(src, acc_out, tgt)
        grpTheGeneratedSeq("string representations")

        target_texts, pred_texts, input_texts = [], [], []
        for seq in target_tokens:
            string = "".join([keyForVal(tok, self.char2hin) for tok in seq])
            target_texts.append(string)
            grpTheGeneratedSeq(string)

        for seq in pred_tokens:
            string = "".join([get_keyAttn(tok, self.char2hin) for tok in seq])
            pred_texts.append(string)
            grpTheGeneratedSeq(string)

        for seq in input_tokens:
            string = "".join([keyForInput(tok, self.char2lat) for tok in seq])
            input_texts.append(string)
        grpTheGeneratedSeq(string)

        self.test_acc.append(torch.tensor(test_acc))
        self.test_loss.append(torch.tensor(test_loss))

        save_outputs_to_csv(input_texts, target_texts, pred_texts)

    #uncomment below code to generate html file

        # if self.counter < 1:
        #     src_str = input_texts[0]
        #     pred_str = pred_texts[0]
        #     attn_np = attn_map[:, 0, :].detach().cpu().numpy()
        #     clipped_attn = attn_np[:len(pred_str), :len(src_str)]
    
        #     # Show attention in notebook (token-to-token)
        #     visualizationOfQ6(
        #         input_sequence=list(src_str),
        #         output_sequence=list(pred_str),
        #         attention_matrix=clipped_attn
        #     )
    
        #     # HTML visualization for full batch
        #     attn_np_list = attn_map.detach().cpu().numpy()  # shape (T, B, S)
        #     attn_per_example = [
        #         attn_np_list[:, i, :].tolist()
        #         for i in range(attn_np_list.shape[1])
        #     ]
        #     generate_connectivity_html(
        #         input_list=[list(x) for x in input_texts],
        #         output_list=[list(x) for x in pred_texts],
        #         attention_list=attn_per_example
        #     )
    
        #     self.counter += 1

    #uncomment below code to generate heat map

        # if self.counter < 1:
        #     plotHeatMap(input_texts, pred_texts, attn_map)
        #     self.counter += 1

        return {'loss': test_loss}

    def on_test_epoch_end(self):
        # Aggregates test loss and accuracy after epoch, clears logs, prints and logs to wandb
        loss_avg = torch.mean(torch.stack(self.test_loss))
        acc_avg  = torch.mean(torch.stack(self.test_acc))
        self.test_loss.clear(); self.test_acc.clear()

        print({
            "Test Loss"     : round(loss_avg.item(), 3),
            "Test Accuracy" : round(acc_avg.item(), 3),
        })
        wandb.log({"Test/Loss": loss_avg, "Test/Accuracy": acc_avg})





# ─────────────────────────────────────────────────────────────────────────────
# 9.  main() – training / validation / test orchestration
# ─────────────────────────────────────────────────────────────────────────────



def main():
    # Load CLI config and select device (GPU/CPU), initialize wandb
    args, device = get_config()

    # Build vocab indices and max sequence lengths from training data
    char_to_idx_latin = getCharToIndex(args)
    charToIndLang     = getCharToIndLang(args)
    max_len_eng       = getMaxLenEng(args)
    max_len_dev       = getMaxLenDev(args)

    # Instantiate the Seq2Seq model with attention using defined parameters
    model = withAttentionSeqToSeq(
        len(char_to_idx_latin) + 2,
        len(charToIndLang) + 2,
        args.hidden_layer_size,
        args.embedding_size,
        args.cell_type,
        args.drop_out,
        args.encoder_layers,
        args.decoder_layers,
        args.bidirectional,
        args.learning_rate,
        max_len_eng,
        charToIndLang,
        char_to_idx_latin
    )

    model.to(device)  # Move model to the selected device (GPU/CPU)

    # Load train, validation, and test datasets as DataLoader objects
    dataloaderTrain, dataloaderVal, dataloaderTest = getDataLoaders(
        device, args, char_to_idx_latin, charToIndLang, max_len_eng, max_len_dev
    )

    # Train and evaluate the model using PyTorch Lightning's Trainer
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator="gpu", devices=1)
    trainer.fit(model=model, train_dataloaders=dataloaderTrain, val_dataloaders=dataloaderVal)
    trainer.test(model, dataloaderTest)

    wandb.finish()  # Finalize Weights & Biases logging

if __name__ == "__main__":
    main()  # Run the pipeline
