import random
from nikola import Nikola

def main():
    nikola = Nikola(depth=4, inputs_per_node=8)
    for epoch in range(200):
        inputs = [random.randint(0, 1) for _ in range(8)]
        target = sum(inputs) % 4
        loss = nikola.meta_train(inputs, target)
        prediction = nikola.forward(inputs)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Inputs: {inputs}, Target: {target}, Prediction: {prediction}, Loss: {loss:.4f}")

if __name__ == "__main__":
    main()
