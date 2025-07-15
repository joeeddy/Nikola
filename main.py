import yaml
import random
from nikola import Nikola
from helpers import generate_inputs, prepare_target, log_activity

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()

    depth = cfg["network"]["depth"]
    inputs_per_node = cfg["network"]["inputs_per_node"]
    epochs = cfg["training"]["epochs"]

    nikola = Nikola(depth=depth, inputs_per_node=inputs_per_node)

    for epoch in range(epochs):
        inputs = generate_inputs(inputs_per_node)
        target = prepare_target(inputs)

        loss = nikola.meta_train(inputs, target)
        prediction = nikola.forward(inputs)

        if epoch % 10 == 0:
            log_activity(epoch, inputs, target, prediction, loss)

if __name__ == "__main__":
    main()
