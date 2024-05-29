import unittest
import torch
import training
import json
import torch.nn as nn
from torch.nn import functional as F
from training import Block, FeedForward, Head, MultiHeadAttention, GPTLanguageModel

# Path for configuration file
CONFIG_FILE = "configurations/config.json"


def load_config():
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)


config = load_config()


class TestModelTraining(unittest.TestCase):
    def setUp(self):
        self.model = GPTLanguageModel(
            vocab_size=100
        )  # Ensure vocab size is correctly set
        self.model.eval()  # Testing in evaluation mode

    def test_batch_processing(self):
        # Create a batch of inputs and targets within the valid range
        batch_x = torch.randint(0, 100, (config["batch_size"], config["block_size"]))
        batch_y = torch.randint(0, 100, (config["batch_size"], config["block_size"]))

        # Run the model to ensure it processes the batch and returns a valid loss tensor
        logits, loss = self.model(batch_x, batch_y)
        self.assertIsNotNone(loss, "Loss should not be None")
        self.assertIsInstance(loss, torch.Tensor, "Loss should be a torch.Tensor")

    def test_training_iteration(self):
        # Simulate a training iteration ensuring the model computes loss and updates weights
        optimizer = torch.optim.Adam(self.model.parameters())
        batch_x = torch.randint(0, 100, (config["batch_size"], config["block_size"]))
        batch_y = torch.randint(0, 100, (config["batch_size"], config["block_size"]))

        optimizer.zero_grad()
        logits, loss = self.model(batch_x, batch_y)
        loss.backward()
        optimizer.step()

        self.assertIsNotNone(loss, "Loss should not be None during training")
        self.assertIsInstance(logits, torch.Tensor, "Logits should be a torch.Tensor")
        self.assertIsInstance(loss, torch.Tensor, "Loss should be a torch.Tensor")


if __name__ == "__main__":
    unittest.main()
