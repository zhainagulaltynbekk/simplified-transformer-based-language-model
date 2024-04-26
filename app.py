import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle  # instead of torch.load and torch.save
from flask import (
    Flask,
    request,
    jsonify,
    Response,
    stream_with_context,
)
from flask_cors import CORS
import time


batch_size = 32  # 64 how many independent sequences will we process in parallel?
block_size = 128  # what is the maximum context length for prediction?
max_iters = 200  # epoch (when user uploads text i can use the data i have to optimize)
eval_interval = 100
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 100
n_embd = 384
n_head = 1  # 8 take way too long
n_layer = 1  # 8
dropout = 0.2

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

app = Flask(__name__)
CORS(app)

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of ints
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take list of integers, output a string


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
            loss = F.cross_entropy(logits, targets)

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


# Initialize your chatbot model
model = GPTLanguageModel(vocab_size)

print("loading model parameters ...")
with open("model/model-01.pk1", "rb") as f:
    model = pickle.load(f)
print("loaded successfully!")
m = model.to("cpu")


# for html
@app.route("/")
def index():
    return render_template("chat.html")


# for html
@app.route("/get-2", methods=["POST"])
def chat2():
    if request.method == "POST":
        msg = request.form["msg"]
        return get_chat_response_without_echo(msg)


# for react buffered way
@app.route("/get", methods=["POST"])
def chat():
    data = request.get_json()  # Get JSON data sent from the client
    msg = data["msg"]
    response_text = get_chat_response_without_echo(msg)
    return jsonify({"message": response_text})


def get_chat_response(text):
    # Generate response from the chatbot model
    context = torch.tensor(encode(text), dtype=torch.long)
    generated_response = decode(
        m.generate(context.unsqueeze(0), max_new_tokens=150)[0].tolist()
    )
    return generated_response


@app.route("/stream", methods=["POST"])
def stream_chat():
    data = request.get_json()  # Get JSON data sent from the client
    msg = data["msg"]

    def generate_stream():
        context = torch.tensor(encode(msg), dtype=torch.long)
        generated_idx = m.generate(context.unsqueeze(0), max_new_tokens=150)[0]

        # Determine where the new text starts by skipping the input context length
        generated_tokens = generated_idx.tolist()[len(context) :]

        # Generate and yield each character of the new text
        for token in generated_tokens:
            character = decode([token])
            yield character
            time.sleep(0.05)  # Delay to simulate typing

    return Response(stream_with_context(generate_stream()), mimetype="text/plain")


def get_chat_response_without_echo(text):
    # Generate response from your chatbot model
    context = torch.tensor(encode(text), dtype=torch.long)
    generated_response = decode(
        m.generate(context.unsqueeze(0), max_new_tokens=150)[0].tolist()
    )
    # Split the generated response by the first occurrence of the input text
    generated_response = generated_response.split(text, 1)[-1].strip()
    return generated_response


if __name__ == "__main__":
    app.run()
