import torch
import torch.nn as nn
import torch.optim as optim

class FractalNode(nn.Module):
    def __init__(self, input_size, hidden_size=16, output_size=4, node_id=0):
        super(FractalNode, self).__init__()
        self.node_id = node_id
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
            nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=0.005)
        self.criterion = nn.CrossEntropyLoss()
        self.connection_weights = {}
        self.activity_log = []
        self.performance = 0.0

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Ensure [1, input_size]
        if x.shape[-1] != self.input_size:
            x = x[:, :self.input_size]
        output = self.network(x)  # Output: [1, output_size]
        if output.dim() == 1:
            output = output.unsqueeze(0)  # Ensure [1, output_size]
        assert output.shape == (1, self.output_size), f"Node {self.node_id} output shape {output.shape}, expected [1, {self.output_size}]"
        return output

    def hebbian_update(self, inputs, outputs, learning_rate=0.02):
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
        self.optimizer.zero_grad()
        output = self.forward(inputs)  # Shape: [1, output_size]
        loss = self.criterion(output, target)  # Expects output: [1, 4], target: [1]
        loss.backward()
        self.optimizer.step()
        self.performance = 0.9 * self.performance + 0.1 * (1 - loss.item())
        return loss.item()
