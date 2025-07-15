import random
import torch

def generate_inputs(n):
    """Generates a random binary input vector of length n."""
    return [random.randint(0, 1) for _ in range(n)]

def prepare_target(inputs):
    """Calculates a target class based on input pattern."""
    return sum(inputs) % 4

def log_activity(epoch, inputs, target, prediction, loss):
    """Prints training progress."""
    print(f"Epoch {epoch} | Inputs: {inputs} | Target: {target} | Prediction: {prediction} | Loss: {loss:.4f}")

def shape_input(x, expected_dim):
    """Ensures input is a 2D float tensor with expected dimensions."""
    x = torch.tensor(x, dtype=torch.float32)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    return x[:, :expected_dim]
