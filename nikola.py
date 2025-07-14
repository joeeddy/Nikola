import torch
import torch.optim as optim
import random
import math
from node import FractalNode

class Nikola:
    def __init__(self, depth=4, inputs_per_node=8):
        self.depth = depth
        self.inputs_per_node = inputs_per_node
        self.nodes = []
        self.connections = []
        self.node_id_counter = 0
        self.temperature = 1.0
        self.create_fractal_network()
        self.meta_optimizer = optim.Adam(self.get_all_parameters(), lr=0.0005)

    def create_fractal_network(self):
        self.nodes = []
        self.connections = []
        self.node_id_counter = 0
        for level in range(self.depth):
            level_nodes = []
            num_nodes = 2 ** level
            for _ in range(num_nodes):
                input_size = self.inputs_per_node if level == 0 else self.inputs_per_node * (2 ** (level - 1))
                node = FractalNode(input_size, hidden_size=16, output_size=4, node_id=self.node_id_counter)
                level_nodes.append(node)
                self.node_id_counter += 1
            self.nodes.append(level_nodes)
        for level in range(1, self.depth):
            for node in self.nodes[level]:
                prev_nodes = [n.node_id for n in self.nodes[level - 1]]
                self.connections.append((node.node_id, prev_nodes[:min(self.inputs_per_node, len(prev_nodes))]))

    def get_all_parameters(self):
        params = []
        for level in self.nodes:
            for node in level:
                params.extend(node.parameters())
        return params

    def forward(self, inputs):
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)  # Shape: [1, input_size]
        layer_outputs = [inputs]
        for level in range(self.depth):
            next_layer_outputs = []
            for node in self.nodes[level]:
                if level == 0:
                    node_inputs = inputs
                else:
                    node_inputs = []
                    for conn in self.connections:
                        if conn[0] == node.node_id:
                            for prev_node_id in conn[1]:
                                for prev_level, nodes in enumerate(self.nodes):
                                    for n in nodes:
                                        if n.node_id == prev_node_id:
                                            output = n.forward(layer_outputs[prev_level])  # Shape: [1, output_size]
                                            node_inputs.append(output)
                    if not node_inputs:
                        node_inputs = torch.zeros(1, node.input_size)
                    else:
                        node_inputs = torch.stack(node_inputs, dim=1).mean(dim=1)  # Mean over inputs, keep [1, input_size]
                        if node_inputs.shape[-1] < node.input_size:
                            node_inputs = torch.cat((node_inputs, torch.zeros(1, node.input_size - node_inputs.shape[-1])), dim=-1)
                        elif node_inputs.shape[-1] > node.input_size:
                            node_inputs = node_inputs[:, :node.input_size]
                output = node.forward(node_inputs)  # Shape: [1, 4]
                node.hebbian_update(node_inputs.squeeze(0), output.squeeze(0))  # Flatten for Hebbian update
                next_layer_outputs.append(output)  # Keep batched: [1, 4]
            layer_outputs.append(torch.cat(next_layer_outputs, dim=0))  # Concatenate to [num_nodes, 4]
        return torch.argmax(layer_outputs[-1][0]).item()  # Take first output for final prediction

    def self_organize(self):
        new_connections = []
        for level in range(1, self.depth):
            for node in self.nodes[level]:
                conn = [c for c in self.connections if c[0] == node.node_id]
                if conn:
                    conn = conn[0]
                    current_conn_ids = conn[1]
                    prev_nodes = [n.node_id for n in self.nodes[level - 1]]
                    new_conn_ids = current_conn_ids
                    if random.random() < math.exp(-1 / self.temperature):
                        available = [nid for nid in prev_nodes if nid not in current_conn_ids]
                        if available:
                            new_conn_ids = random.sample(prev_nodes, min(self.inputs_per_node, len(prev_nodes)))
                    current_score = sum(n.connection_weights.get(i, 0) for i in current_conn_ids for n in self.nodes[level - 1])
                    new_score = sum(n.connection_weights.get(i, 0) for i in new_conn_ids for n in self.nodes[level - 1])
                    if new_score > current_score or random.random() < math.exp((new_score - current_score) / self.temperature):
                        new_connections.append((node.node_id, new_conn_ids))
                    else:
                        new_connections.append(conn)
        self.connections = new_connections
        self.temperature *= 0.99
        for level in range(self.depth):
            for node in self.nodes[level]:
                if len(node.activity_log) > 20 and sum(node.activity_log[-20:]) / 20 < 0.2:
                    self.nodes[level] = [n for n in self.nodes[level] if n.node_id != node.node_id]
                    self.connections = [c for c in self.connections if c[0] != node.node_id]
                elif node.performance > 0.9 and random.random() < 0.1:
                    new_node = FractalNode(node.input_size, hidden_size=16, output_size=4, node_id=self.node_id_counter)
                    self.node_id_counter += 1
                    self.nodes[level].append(new_node)
                    if level < self.depth - 1:
                        self.connections.append((new_node.node_id, [n.node_id for n in self.nodes[level]]))

    def meta_train(self, inputs, target):
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)  # Shape: [1, input_size]
        target = torch.tensor([target], dtype=torch.long)  # Shape: [1]
        original_params = [p.clone().detach() for p in self.get_all_parameters()]
        fast_weights = [p.clone().detach().requires_grad_(True) for p in self.get_all_parameters()]
        loss = 0
        for level in self.nodes:
            for node in level:
                output = node.forward(inputs)  # Shape: [1, 4]
                loss += node.train_step(inputs, target)
        fast_optimizer = optim.Adam(fast_weights, lr=0.01)
        fast_optimizer.zero_grad()
        for level in self.nodes:
            for node in level:
                output = node.forward(inputs)  # Shape: [1, 4]
                loss = node.criterion(output, target)
                loss.backward()
        fast_optimizer.step()
        self.meta_optimizer.zero_grad()
        meta_loss = 0
        for level in self.nodes:
            for node in level:
                output = node.forward(inputs)  # Shape: [1, 4]
                meta_loss += node.criterion(output, target)
        meta_loss.backward()
        self.meta_optimizer.step()
        self.self_organize()
        return meta_loss.item()
