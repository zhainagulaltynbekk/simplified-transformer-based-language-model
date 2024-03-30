# Simplified-Transformer-Based-Language-Model
TRAINING:
1. install necessary libraries (in a venv suggested)
    `pip install matplotlib numpy pylzma ipykernel jupyter tqdm Flask`
    `pip install torch --index-url https://download.pytorch.org/whl/cu118`
    `python -m ipykernel install --user --name=cuda --display-name "cuda-gpt"`

2. split your data to vaidation and training and make your unique vocab list with data-extract.py `python data-extract.py`
3. train your data `python training.py`
4. chatbot `python chatbot.py`