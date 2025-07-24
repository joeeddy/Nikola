import torch
import torch.optim as optim
from topology import create_fractal_network, self_organize
from helpers import shape_input

class Nikola:
    def __init__(self, depth, inputs_per_node, hidden_size=16, output_size=4, learning_rate_meta=0.0005):
        self.depth = depth
        self.inputs_per_node = inputs_per_node
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.temperature = 1.0

        self.nodes, self.connections, self.node_id_counter = create_fractal_network(
            depth, inputs_per_node, hidden_size, output_size
        )
        self.meta_optimizer = optim.Adam(self.get_all_parameters(), lr=learning_rate_meta)

    def collect_inputs(self, node_id, layer_outputs, level):
        # Simple placeholder: just return the previous layer's outputs
        return layer_outputs[level]

    def get_all_parameters(self):
        params = []
        for layer in self.nodes:
            for node in layer:
                params.extend(node.parameters())
        return params

    def forward(self, inputs):
        # Ensure inputs are correctly shaped
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
                # Squeeze batch dimension for hebbian_update
                node.hebbian_update(node_inputs.squeeze(0), output.squeeze(0))
                next_layer_outputs.append(output)
            # Concatenate outputs along batch dimension
            layer_outputs.append(torch.cat(next_layer_outputs, dim=0))
        
        # Return the entire batch of final outputs
        return layer_outputs[-1]

    def meta_train(self, inputs, target):
        # Prepare inputs and target
        inputs = shape_input(inputs, self.inputs_per_node)
        target = torch.tensor([target], dtype=torch.long)

        layer_outputs = [inputs]
        total_loss = 0

        # Forward pass through each layer
        for level in range(self.depth):
            next_layer_outputs = []
            for node in self.nodes[level]:
                if level == 0:
                    node_inputs = inputs
                else:
                    node_inputs = self.collect_inputs(node.node_id, layer_outputs, level - 1)
                output = node.forward(node_inputs)
                loss = node.train_step(node_inputs, target)
                # Accumulate loss
                total_loss += loss
                next_layer_outputs.append(output)
            # Concatenate for next layer
            layer_outputs.append(torch.cat(next_layer_outputs, dim=0))
        
        # Backpropagation for meta-optimizer
        self.meta_optimizer.zero_grad()
        # Compute meta loss, e.g., mean over nodes
        meta_loss = total_loss / (self.depth * len(self.nodes[-1]))
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()
          
