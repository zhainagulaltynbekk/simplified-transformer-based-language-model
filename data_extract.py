import os
import lzma
from tqdm import tqdm
import requests
import PyPDF2
import torch


# only url with pdf or txt files should be given
def download_file_from_url(url, save_path):
    """Download a file from a URL to a specified local path"""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
    else:
        print(f"Failed to download {url}")


def download_files_from_urls(file_urls):
    for url in file_urls:
        filename = url.split("/")[-1]
        download_file_from_url(url, os.path.join("data", filename))


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Assuming 'data' directory contains both PDF and TXT files downloaded previously
def process_files(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith((".pdf", ".txt"))]
    text_output_path = "data/combined_text.txt"
    with open(text_output_path, "w", encoding="utf-8") as outfile:
        for filename in files:
            file_path = os.path.join(folder_path, filename)
            if filename.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as infile:
                    text = infile.read()
            elif filename.endswith(".pdf"):
                text = extract_text_from_pdf(file_path)
            outfile.write(text)


class FileProcessor:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.vocab = set()
        self.characters = []
        self.bigrams = {}

    def xz_files_in_dir(self):
        """List .xz or .txt files in a directory."""
        files = []
        for filename in os.listdir(self.folder_path):
            if (
                filename.endswith(".xz") or filename.endswith(".txt")
            ) and os.path.isfile(os.path.join(self.folder_path, filename)):
                files.append(filename)
        return files

    def process_text_files(
        self, output_file_train, output_file_val, vocab_file, train_file_percentage
    ):
        """Process text and .xz files, split into training and validation, update vocabulary."""
        files = self.xz_files_in_dir()
        total_files = len(files)
        split_index = int(total_files * train_file_percentage)  # 90% for training
        files_train = files[:split_index]
        files_val = files[split_index:]

        # it reads the content using lzma.open to decompress the data (assuming it's compressed with LZMA),
        # then writes the content to the output file. Meanwhile, it keeps track of unique characters encountered in the vocab set
        # process the training files
        with open(output_file_train, "w", encoding="utf-8") as outfile:
            for filename in tqdm(files_train, total=len(files_train)):
                file_path = os.path.join(self.folder_path, filename)
                with (
                    open(file_path, "rt", encoding="utf-8")
                    if filename.endswith(".txt")
                    else lzma.open(file_path, "rt", encoding="utf-8")
                ) as infile:
                    text = infile.read()
                    outfile.write(text)
                    self.vocab.update(set(text))
                    self.characters = list(text)

        with open(output_file_val, "w", encoding="utf-8") as outfile:
            for filename in tqdm(files_val, total=len(files_val)):
                file_path = os.path.join(self.folder_path, filename)
                with (
                    open(file_path, "rt", encoding="utf-8")
                    if filename.endswith(".txt")
                    else lzma.open(file_path, "rt", encoding="utf-8")
                ) as infile:
                    text = infile.read()
                    outfile.write(text)
                    self.vocab.update(set(text))
                    self.characters.extend(list(text))

        with open(vocab_file, "w", encoding="utf-8") as vfile:
            for char in self.vocab:
                vfile.write(char + "\n")

        # bigram
        self.compute_bigrams()

    def compute_bigrams(self):
        """Compute bigrams from the accumulated character list."""
        characters_set = set(self.characters)
        characters_set_len = len(characters_set)
        for ch1, ch2 in zip(self.characters, self.characters[1:]):
            bigram = (ch1, ch2)
            self.bigrams[bigram] = self.bigrams.get(bigram, 0) + 1

        bigrams_len = len(sorted(self.bigrams.items(), key=lambda kv: -kv[1]))
        print(bigrams_len)

        N = torch.zeros((characters_set_len, characters_set_len), dtype=torch.int32)

        sorted_characters_list = sorted(list(characters_set))
        bigram_stoi = {s: i for i, s in enumerate(sorted_characters_list)}
        bigram_itos = {i: s for s, i in bigram_stoi.items()}

        for ch1, ch2 in zip(self.characters, self.characters[1:]):
            if (
                ch1 in bigram_stoi and ch2 in bigram_stoi
            ):  # Check if both characters are in the vocabulary
                ix1 = bigram_stoi[ch1]
                ix2 = bigram_stoi[ch2]
                if ix1 >= len(sorted_characters_list) or ix2 >= len(
                    sorted_characters_list
                ):
                    print(
                        f"Index out of bounds for characters: {ch1}, {ch2}, Indices: {ix1}, {ix2}"
                    )
                else:
                    N[ix1, ix2] += 1
            else:
                print(
                    f"Unseen characters: {ch1}, {ch2}"
                )  # Debugging print for unseen characters

        self.bigrams = {
            (bigram_itos[i], bigram_itos[j]): N[i, j].item()
            for i in range(len(sorted_characters_list))
            for j in range(len(sorted_characters_list))
        }

    def get_bigrams(self):
        """Returns the computed bigram data."""
        bigram_list = [
            {"bigram": f"{ch1}{ch2}", "count": count}
            for (ch1, ch2), count in self.bigrams.items()
            if count > 0
        ]
        return bigram_list
