import io
import sys
import torch
import json
import pickle  # instead of torch.load and torch.save
from flask import (
    Flask,
    request,
    jsonify,
    Response,
    stream_with_context,
    send_from_directory,
)
from flask_cors import CORS
import os
import time
import numpy as np
from werkzeug.utils import secure_filename
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import data_extract
import training
from training import Block, FeedForward, Head, MultiHeadAttention, GPTLanguageModel
import seaborn
import pandas as pd
import subprocess

# Path for configuration file
CONFIG_FILE = "configurations/config.json"
MODEL_DIR = "model/new-model/"
# Ensure the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Define the path where uploaded files will be stored
UPLOAD_FOLDER = "data/files/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
IMAGE_DIR = "plt_images/"
os.makedirs(IMAGE_DIR, exist_ok=True)


def load_config():
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)


def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)


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

app = Flask(__name__)
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})
# CORS(app, resources={r"/model-train/*": {"origins": "http://localhost:3000"}})

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of ints
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take list of integers, output a string

# Initialize your chatbot model
try:
    print("loading model parameters ...")
    with open(config["model_path"], "rb") as f:
        model = pickle.load(f)
    print("loaded successfully!")
except FileNotFoundError:
    print("Model file not found, initializing new model!")
    model = GPTLanguageModel(vocab_size)

m = model.to(config["device"])


# Data Preperation route
@app.route("/upload-files", methods=["POST"])
def handle_files():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files provided"}), 400
    try:
        for file in files:
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))

        # Process the files using processor in data_extract.pyss
        train_file_percentage = 0.9
        val_file_percentage = 1 - train_file_percentage
        output_file_train = "data/train_test.txt"
        output_file_val = "data/val_test.txt"
        vocab_file = "data/vocab_test.txt"
        app.logger.info("Starting file processing...")
        processor = data_extract.FileProcessor(UPLOAD_FOLDER)
        processor.process_text_files(
            output_file_train, output_file_val, vocab_file, train_file_percentage
        )

        bigrams = processor.get_bigrams()
        app.logger.info("File processing completed successfully.")
        # Prepare the data to send back
        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab = f.read()
        with open(output_file_train, "r", encoding="utf-8") as f:
            train_length = len(f.read())
        with open(output_file_val, "r", encoding="utf-8") as f:
            val_length = len(f.read())

        return (
            jsonify(
                {
                    "message": f"{len(files)} files processed successfully",
                    "upload_folder": UPLOAD_FOLDER,
                    "train_file_percentage": train_file_percentage * 100,
                    "val_file_percentage": val_file_percentage * 100,
                    "ouput_file_val": output_file_val,
                    "output_file_train": output_file_train,
                    "vocab_file": vocab_file,
                    "vocab": list(set(vocab)),
                    "uploadedFiles": [file.filename for file in files],
                    "vocabLength": len(set(vocab)),
                    "trainLength": train_length,
                    "valLength": val_length,
                    "bigrams": bigrams,
                    "bigram_len": len(bigrams),
                }
            ),
            200,
        )
    except Exception as e:
        app.logger.error(f"Error processing files: {e}")
        return jsonify({"error": str(e)}), 500


# Train route
@app.route("/submit-form", methods=["POST"])
def handle_form_submission():
    # Load existing configuration
    load_config()

    # Process text fields and update config
    form_data = {key: value for key, value in request.form.items()}
    for key, value in form_data.items():
        if key in config:  # Only update if the key exists in the config
            try:
                # Convert to appropriate type based on current config type
                config[key] = type(config[key])(value)
            except ValueError:
                continue  # Skip if conversion fails, you could log this or handle differently

    # Handle file upload
    file = request.files.get("file")
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(MODEL_DIR, filename)
        file.save(file_path)
        config["model_path"] = file_path  # Update config to new model path
        print(f"File saved to {file_path}")
    else:
        config["model_path"] = "model/model-01.pk1"  # No file uploaded, use default

    # Save updated configuration
    save_config(config)

    # Prepare a response
    response = {
        "message": "Form submitted successfully!",
        "receivedData": form_data,
        "fileSaved": filename if file else "No file uploaded",
        "configUpdated": config,
    }
    return jsonify(response), 200


# Progress route
@app.route("/model-train")
def model_train_endpoint():
    def generate():
        process = subprocess.Popen(
            ["python", "training.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in iter(process.stdout.readline, ""):
            yield f"data:{line}\n\n"
            sys.stdout.flush()

        process.stdout.close()
        process.wait()
        yield "data:done\n\n"  # Indicate the end of the stream

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


# Progress MAP image display
@app.route("/images/<filename>")
def serve_image(filename):
    return send_from_directory(IMAGE_DIR, filename)


# Chat route
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


if __name__ == "__main__":
    app.run()
