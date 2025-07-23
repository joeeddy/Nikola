import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class FractalNode(nn.Module):
    def __init__(self, input_size, hidden_size=16, output_size=4, node_id=0):
        super(FractalNode, self).__init__()
        self.node_id = node_id
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Define neural network layers
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
            nn.Softmax(dim=-1)
        )

        # Optimizer and loss criterion
        self.optimizer = optim.Adam(self.parameters(), lr=0.005)
        self.criterion = nn.CrossEntropyLoss()

        # Connection weights (for Hebbian update)
        self.connection_weights = {}
        # Activity log for monitoring
        self.activity_log = []
        # Performance metric
        self.performance = 0.0

        # To store output after forward pass (used externally)
        self.output = None

    def forward(self, x):
        """
        Forward pass through the network.
        """
        # Convert input if not tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        # Ensure batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)
        # Slice input if needed
        if x.shape[-1] != self.input_size:
            x = x[:, :self.input_size]
        output = self.network(x)  # Output shape: [batch_size, output_size]

        # Debug prints
        print(f"Node {self.node_id} - input shape: {x.shape}")
        print(f"Node {self.node_id} - output shape: {output.shape}")

        # Ensure output is batch size 1
if output.dim() == 1:
    output = output.unsqueeze(0)
# Modify the assertion to accept batch size > 1
assert output.shape[1] == self.output_size, \
    f"Node {self.node_id} output shape {output.shape}, expected batch size 1 with shape [(1, {self.output_size})]"
self.output = output  # Store for external access
return output

    def hebbian_update(self, inputs, outputs, learning_rate=0.02):
        """
        Update connection weights based on correlation between inputs and outputs.
        """
        inputs = torch.tensor(inputs, dtype=torch.float32)
        outputs = torch.tensor(outputs, dtype=torch.float32)
        correlation = torch.mean(inputs * outputs).item()
        for i in range(len(inputs)):
            if i in self.connection_weights:
                self.connection_weights[i] += learning_rate * correlation
                self.connection_weights[i] = max(0.2, min(1.0, self.connection_weights[i]))
            else:
                self.connection_weights[i] = 0.5
        self.activity_log.append(correlation)
        if len(self.activity_log) > 20:
            self.activity_log.pop(0)

    def train_step(self, inputs, target):
        """
        Compute the loss, but do NOT perform backpropagation or optimizer update.
        Returns the loss value.
        """
        # Ensure inputs are tensor
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float32)
        # Forward pass
        output = self.forward(inputs)
        # Compute loss
        loss = self.criterion(output, target)
        return loss

# Optional: You might want to add a method to perform actual training update
# def update_parameters(self, loss):
#     self.optimizer.zero_grad()
#     loss.backward()
#     self.optimizer.step()

def get_node_activity_map():
    # Simulate a 10x10 matrix of activity levels (for visualization)
    return np.random.rand(10, 10)
