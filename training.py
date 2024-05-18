import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle  # instead of torch.load and torch.save
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn
import pandas as pd
import matplotlib.pyplot as plt

# hyperparameters
# parser = argparse.ArgumentParser(description="For argument values!")
# # Here we add an argument to the parser, specifying the expected type, a help message, etc.
# parser.add_argument(
#     "-batch_size", type=str, required=True, help="Please provide a batch_size"
# )
# args = parser.parse_args()
# # Now we can use the argument value in our program.
# print(f"batch_size: {args.batch_size}")
# batch_size = int(
#     args.batch_size
# )
batch_size = 32  # 64 how many independent sequences will we process in parallel?
block_size = 128  # what is the maximum context length for prediction? length of a chunk
max_iters = 10  # epoch (when user uploads text i can use the data i have to optimize)
eval_interval = 100
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 1
n_embd = 384
n_head = 8  # 8 take way too long
n_layer = 8  # 8
dropout = 0.2

print(device)
torch.manual_seed(
    1337
)  # when you set a random seed, it means that each time you run your program, you will get the same sequence of random numbers.

chars = ""
# with open ('wizard_of_oz.txt', 'r', encoding='utf-8') as f:
with open(
    "data/vocab.txt", "r", encoding="utf-8"
) as f:  # when we open huge files we can not open them in RAM at once, so we use unique set of chars instead
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Tokenizer
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of ints
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take list of integers, output a string

# tensors(different type of data types) instead of arrays
# data = torch.tensor(encode(text), dtype=torch.long) # we take data as a tensor and make sure that the dtype is long


# memory map for using small snippets of text from a single file of any size
def get_random_chunk(split):  # split indicates whether to load data for train or val
    filename = (
        "data/train_split.txt" if split == "train" else "data/val_split.txt"
    )  # takes appropriate file accroding to the split
    with open(filename, "rb") as f:
        with mmap.mmap(
            f.fileno(), 0, access=mmap.ACCESS_READ
        ) as mm:  # opens the file in a binary mode and creates a memory-mapped file object (mmap) to efficiently access chunks of data without loading the entire file into memory
            # Determines the file size and a random position to start reading
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - block_size * batch_size)

            # Seek to the random position and read the block of text
            mm.seek(start_pos)
            block = mm.read(block_size * batch_size - 1)

            # Decode the block to a string, ignoring any invalid byte sequences
            decoded_block = block.decode("utf-8", errors="ignore").replace("\r", "")

            # Train and test splits
            data = torch.tensor(encode(decoded_block), dtype=torch.long)

    return data


# data loading
def get_batch(split):
    data = get_random_chunk(split)  # gets a chunk of data for the specific split
    ix = torch.randint(
        len(data) - block_size, (batch_size,)
    )  # generates random indices ix for selecting batch from data
    # creates tensors x and y representing input and target sequences, respectively, for the language model.
    # x is a stack of subsequences from the data, each of length block_size.
    # y is a stack of subsequences that follow the corresponding subsequences in x.
    # moves the tensors to the specified device (CPU or GPU).
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()  # PyTorch will make sure not to use gradients here
def estimate_loss(model):
    out = {}
    model.eval()  # puts the model on an evaluation mode
    for split in ["train", "val"]:
        losses = torch.zeros(
            eval_iters
        )  # stores the losses obtained in multiple iterations
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # puts the model on a training mode
    return out


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat(
            [h(x) for h in self.heads], dim=-1
        )  # (B,T,F) feature dimension -> (B, T, [h1, h1, h1, h1, h2, h2, h2, h2, h3, h3, h3, h3])
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(index)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape  # batch time and a channel is the vocabulary size
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            # loss = F.mse_loss(
            #     logits.argmax(dim=1).type(torch.float), targets.type(torch.float)
            # )
            loss = F.cross_entropy(logits, targets)
            # loss = torch.sqrt(
            #     torch.mean(
            #         (logits.argmax(dim=1).type(torch.float) - targets.type(torch.float))
            #         ** 2
            #     )
            # )

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


def main():
    model = GPTLanguageModel(vocab_size)
    # if you don't have your model make sure to comment this part before you create your model!
    # with this we will be able to train our model  multiple times
    print("loading model parameters ...")
    with open("model/model-01.pk1", "rb") as f:
        model = pickle.load(f)
    print("loaded successfully!")
    m = model.to(device)

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        # sample a batch of data
        xb, yb = get_batch("train")

        # evaluate the loss
        logits, loss = model.forward(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()  # neuron get updated ????
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_iters == 0:
            losses = estimate_loss(model)
            print(
                f"step {iter}, train loss {losses['train']:.3f}, validation loss {losses['val']:.3f}, model loss {loss:.3f}"
            )
            xt, yt = get_batch("val")
            y_predicted = model.generate(xt, max_new_tokens=2)[:, xt.shape[1] :]
            # y_predicted = decode(y_predicted)
            # yt = decode(y)
            accuracy_ = accuracy_score(yt[:, :2].flatten(), y_predicted.flatten())
            print(accuracy_)
            accuracy = np.sum((yt[:, :2] == y_predicted).numpy()) / (len(yt) * 2)
            print(f"LOG: accuracy: {accuracy}")

        xt, yt = get_batch("val")
        y_predicted = model.generate(xt, max_new_tokens=2)[:, xt.shape[1] :]

        cm = confusion_matrix(yt[:, :2].flatten(), y_predicted.flatten())
        labels = np.unique(np.concatenate((yt[:, :2], y_predicted)))
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        cm_plot = seaborn.heatmap(cm_df, annot=True, cmap="Blues")
        cm_plot.set_xlabel("Predicted Values")
        cm_plot.set_ylabel("Actual Values")
        cm_plot.set_title("Confusion Matrix", size=16)
        # plt.show()

    print(loss.item())

    #  it serializes the trained model and writes it to the file
    with open("model/model-01.pk1", "wb") as f:
        pickle.dump(model, f)  # dump = save
    print("model saved")

    # generate from the models
    def generate_text():
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated_chars = decode(m.generate(context, max_new_tokens=500)[0].tolist())
        return generated_chars

    print(generate_text())


if __name__ == "__main__":
    main()
