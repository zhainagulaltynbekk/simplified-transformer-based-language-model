from training import GPTLanguageModel, Block, Head, MultiHeadAttention, FeedForward
import torch
import json
import pickle
import sys


# Path for configuration file
CONFIG_FILE = "configurations/config.json"


def load_config():
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)


config = load_config()


chars = ""
# with open ('wizard_of_oz.txt', 'r', encoding='utf-8') as f:
with open(
    "data/vocab.txt", "r", encoding="utf-8"
) as f:  # when we open huge files we can not open them in RAM at once, so we use unique set of chars instead
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of ints
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take list of integers, output a string

# Initialize your chatbot model


def main():
    try:
        # print("loading model parameters ...")
        with open(config["model_path"], "rb") as f:
            model = pickle.load(f)
        # print("loaded successfully!")
    except FileNotFoundError:
        print("Model file not found, initializing new model!")
        model = GPTLanguageModel(vocab_size)
    m = model.to(config["device"])

    msg = sys.argv[1]

    context = torch.tensor(encode(msg), dtype=torch.long)
    for generated_idx in m.generate_token_by_token(
        context.unsqueeze(0), max_new_tokens=150
    ):
        generated_idx = generated_idx[0]
        # Determine where the new text starts by skipping the input context length
        generated_tokens = generated_idx.tolist()

        # Generate and yield each character of the new text
        for token in generated_tokens:
            character = decode([token])
            print(character)


if __name__ == "__main__":
    main()
