import os
import lzma
from tqdm import tqdm
import requests
import PyPDF2


# only url with pdf or txt files should be given
def download_file_from_url(url, save_path):
    """ Download a file from a URL to a specified local path """
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
    else:
        print(f"Failed to download {url}")


# Example of downloading multiple files
# file_urls = [
#     '',
#     ''
# ]
def download_files_from_urls(file_urls):
    for url in file_urls:
        filename = url.split('/')[-1]
        download_file_from_url(url, os.path.join('data', filename))


def extract_text_from_pdf(pdf_path):
    """ Extract text from a PDF file """
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Assuming 'data' directory contains both PDF and TXT files downloaded previously
def process_files(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith(('.pdf', '.txt'))]
    text_output_path = 'data/combined_text.txt'
    with open(text_output_path, 'w', encoding='utf-8') as outfile:
        for filename in files:
            file_path = os.path.join(folder_path, filename)
            if filename.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as infile:
                    text = infile.read()
            elif filename.endswith('.pdf'):
                text = extract_text_from_pdf(file_path)
            outfile.write(text)

# process_files('data')

# takes a directory path as input and returns a list of filenames for files with a ".xz" ot "txt" extension in that directory
def xz_files_in_dir(directory):
    files = []
    for filename in os.listdir(directory):
        if (filename.endswith(".xz") or filename.endswith(".txt")) and os.path.isfile(
            os.path.join(directory, filename)
        ):
            files.append(filename)
    return files


# folder_path = "C:/Users/ealtzha/OneDrive - Ericsson/Desktop/THESIS/openwebtext/openwebtext"  # where our xz files are located
folder_path = (
    "C:\\Users\\ealtzha\\OneDrive - Ericsson\\Desktop\\THESIS\\data-for-model-training"
)

output_file_train = "data/train_split.txt"
output_file_val = "data/val_split.txt"
vocab_file = "data/vocab.txt"  # file where we want to save our vocabulary,
# from the file that we are reading each time when we get the new character we gonna push it into this file
# vocab_file simply contains all the unique characters in our file

files = xz_files_in_dir(folder_path)
total_files = len(files)

# Calculate the split indices
split_index = int(total_files * 0.9)  # 90% for training
files_train = files[:split_index]
print(f"Files for the training: {files_train}")
files_val = files[split_index:]
print(f"Files for the validation: {files_val}")

# Split into train and validation data (we don't want out AI to generate exact same data, but to generate something like it)
# process the training and validation seperately
vocab = set()

# it reads the content using lzma.open to decompress the data (assuming it's compressed with LZMA),
# then writes the content to the output file. Meanwhile, it keeps track of unique characters encountered in the vocab set
# process the training files
with open(output_file_train, "w", encoding="utf-8") as outfile:
    for filename in tqdm(files_train, total=len(files_train)):
        file_path = os.path.join(folder_path, filename)
        with (
            open(file_path, "rt", encoding="utf-8")
            if filename.endswith(".txt")
            else lzma.open(file_path, "rt", encoding="utf-8")
        ) as infile:
            text = infile.read()
            outfile.write(text)
            characters = set(text)
# vocab.update(characters)  # uncomment when you want to update your vocab.txt

# process the validation files
with open(output_file_val, "w", encoding="utf-8") as outfile:
    for filename in tqdm(files_val, total=len(files_val)):
        file_path = os.path.join(folder_path, filename)
        with (
            open(file_path, "rt", encoding="utf-8")
            if filename.endswith(".txt")
            else lzma.open(file_path, "rt", encoding="utf-8")
        ) as infile:
            text = infile.read()
            outfile.write(text)
            characters = set(text)
vocab.update(characters)

# the unique characters in the vocabulary are then written to a file specified
# by vocab_file (presumably as a list of characters, one per line)
with open(vocab_file, "w", encoding="utf-8") as vfile:
    for char in vocab:
        vfile.write(char + "\n")
