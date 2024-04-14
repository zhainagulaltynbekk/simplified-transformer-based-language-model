# import numpy as np
import matplotlib.pyplot as plt
import torch

# Load vocabulary
with open("data/vocab_test.txt", "r", encoding="utf-8") as file:
    # Each character is on its own line
    characters = file.read().strip().split("\n")
# Ensure uniqueness and sort characters for consistent indexing
unique_chars = sorted(set(characters))
print(len(characters))
print(len(unique_chars))
print(unique_chars)

b = {}
for ch1, ch2 in zip(characters, characters[1:]):
    bigram = (ch1, ch2)
    b[bigram] = b.get(bigram, 0) + 1

sorted_bigram_counts = sorted(b.items(), key=lambda kv: -kv[1])
print(sorted_bigram_counts)

# Initialize N matrix for storing bigram frequencies
# The shape of N is (len(unique_chars), len(unique_chars))
N = torch.zeros((len(unique_chars), len(unique_chars)), dtype=torch.int32)

# Create mapping from characters to integers and back
stoi = {s: i for i, s in enumerate(unique_chars)}
itos = {i: s for s, i in stoi.items()}

# Calculate bigram frequencies
for ch1, ch2 in zip(characters, characters[1:]):
    if ch1 in stoi and ch2 in stoi:
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

# Visualization
plt.figure(figsize=(50, 50))
plt.imshow(N, cmap="Blues")
for i in range(len(unique_chars)):
    for j in range(len(unique_chars)):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color="gray")
        plt.text(j, i, str(N[i, j]), ha="center", va="top", color="gray")
plt.axis("off")

# Save the figure or display it
# plt.savefig("bigram_visualization.png")
plt.show()
