import torch
import torch.optim as optim
from topology import create_fractal_network
from helpers import shape_input

class Nikola:
    def __init__(self, depth, inputs_per_node, hidden_size=16, output_size=4, learning_rate_meta=0.0005):
        self.depth = depth
        self.inputs_per_node = inputs_per_node
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.temperature = 1.0

        # Create network topology
        self.nodes, self.connections, self.node_id_counter = create_fractal_network(
            depth, inputs_per_node, hidden_size, output_size
        )

        # Create a list of all node parameters for optimizer
        self.node_parameters = []
        for layer in self.nodes:
            for node in layer:
                self.node_parameters.extend(node.parameters())

        # Meta optimizer for all nodes
        self.meta_optimizer = optim.Adam(self.node_parameters, lr=learning_rate_meta)

    def collect_inputs(self, node_id, layer_outputs, level):
        # Return outputs from previous layer
        return layer_outputs[level]

    def get_all_parameters(self):
        # Return list of all node parameters
        params = []
        for layer in self.nodes:
            for node in layer:
                params.extend(node.parameters())
        return params

    def forward(self, inputs):
        # Ensure inputs are shaped properly
        inputs = shape_input(inputs, self.inputs_per_node)
        layer_outputs = [inputs]

        for level in range(self.depth):
            next_layer_outputs = []
            for node in self.nodes[level]:
                if level == 0:
                    node_inputs = inputs
                else:
                    node_inputs = self.collect_inputs(node.node_id, layer_outputs, level - 1)
                output = node.forward(node_inputs)
                # Update Hebbian connections
                node.hebbian_update(node_inputs.squeeze(0), output.squeeze(0))
                next_layer_outputs.append(output)
            # Concatenate outputs for next layer
            layer_outputs.append(torch.cat(next_layer_outputs, dim=0))
        return layer_outputs[-1]

    def meta_train(self, inputs, target):
        # Prepare inputs
        inputs = shape_input(inputs, self.inputs_per_node)
        batch_size = inputs.shape[0]
        # Expand target to batch size
        target = torch.tensor([target] * batch_size, dtype=torch.long)

        # Assert batch sizes match
        assert inputs.shape[0] == target.shape[0], f"Input and target batch size mismatch: {inputs.shape} vs {target.shape}"

        layer_outputs = [inputs]
        total_loss = 0

        # Forward through layers
        for level in range(self.depth):
            next_layer_outputs = []
            for node in self.nodes[level]:
                if level == 0:
                    node_inputs = inputs
                else:
                    node_inputs = self.collect_inputs(node.node_id, layer_outputs, level - 1)
                output = node.forward(node_inputs)
                # Train node
                loss = node.train(node_inputs, target)
                total_loss += loss
                next_layer_outputs.append(output)
            # Concatenate for next layer
            layer_outputs.append(torch.cat(next_layer_outputs, dim=0))

        # Compute meta loss
        self.meta_optimizer.zero_grad()
        meta_loss = total_loss / (self.depth * len(self.nodes[-1]))
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()
