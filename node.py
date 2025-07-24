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

        # Optimizer for this node
        self.optimizer = optim.Adam(self.parameters(), lr=0.005)

        # Loss criterion
        self.criterion = nn.CrossEntropyLoss()

        # Connection weights for Hebbian update (optional, example)
        self.connection_weights = {}  # key: input index, value: weight
        # Activity log for monitoring
        self.activity_log = []
        # Performance metric (could be accuracy or other)
        self.performance = 0.0

        # Store output after forward pass (used externally)
        self.output = None

    def forward(self, x):
        """
        Forward pass through the network.
        """
        # Ensure input is tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        # Ensure batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Slice input if needed
        if x.shape[-1] != self.input_size:
            x = x[:, :self.input_size]

        output = self.network(x)

        # Debug prints (optional)
        # print(f"Node {self.node_id} - input shape: {x.shape}")
        # print(f"Node {self.node_id} - output shape: {output.shape}")

        self.output = output
        return output

    def hebbian_update(self, inputs, outputs, learning_rate=0.02):
        """
        Update connection weights based on correlation between inputs and outputs.
        """
        # Convert to tensors if needed
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float32)
        if not isinstance(outputs, torch.Tensor):
            outputs = torch.tensor(outputs, dtype=torch.float32)

        # Calculate correlation (simplified example)
        correlation = torch.mean(inputs * outputs).item()

        # Update connection weights (example logic)
        for i in range(len(inputs)):
            if i in self.connection_weights:
                self.connection_weights[i] += learning_rate * correlation
                # Clamp weights
                self.connection_weights[i] = max(0.2, min(1.0, self.connection_weights[i]))
            else:
                self.connection_weights[i] = 0.5

        # Log activity
        self.activity_log.append(correlation)
        if len(self.activity_log) > 20:
            self.activity_log.pop(0)

    def train(self, inputs, target):
        """
        Perform a training step: zero grad, forward, compute loss, backward, optimizer step.
        Returns the loss value.
        """
        self.optimizer.zero_grad()

        # Forward pass
        output = self.forward(inputs)

        # Compute loss
        loss = self.criterion(output, target)

        # Backward and optimize
        loss.backward()
        self.optimizer.step()

        return loss.item()
