import os, lzma
from tqdm import tqdm


# takes a directory path as input and returns a list of filenames for files with a ".xz" extension in that directory
def xz_files_in_dir(directory):
    files = []
    for filename in os.listdir(directory):
        if filename.endswith(".xz") and os.path.isfile(
            os.path.join(directory, filename)
        ):
            files.append(filename)
    return files


folder_path = "C:/Users/ealtzha/OneDrive - Ericsson/Desktop/THESIS/openwebtext/openwebtext"  # where our xz files are located
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
files_val = files[split_index:]

# Split into train and validation data (we don't want out AI to generate exact same data, but to generate something like it)
# process the training and validation seperately
vocab = set()

# it reads the content using lzma.open to decompress the data (assuming it's compressed with LZMA),
# then writes the content to the output file. Meanwhile, it keeps track of unique characters encountered in the vocab set
# process the training files
with open(output_file_train, "w", encoding="utf-8") as outfile:
    for filename in tqdm(files_train, total=len(files_train)):
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path, "rt", encoding="utf-8") as infile:
            text = infile.read()
            outfile.write(text)
            characters = set(text)
            vocab.update(characters)

# process the validation files
with open(output_file_val, "w", encoding="utf-8") as outfile:
    for filename in tqdm(files_val, total=len(files_val)):
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path, "rt", encoding="utf-8") as infile:
            text = infile.read()
            outfile.write(text)
            characters = set(text)
            vocab.update(characters)

# the unique characters in the vocabulary are then written to a file specified
# by vocab_file (presumably as a list of characters, one per line)
with open(vocab_file, "w", encoding="utf-8") as vfile:
    for char in vocab:
        vfile.write(char + "\n")
