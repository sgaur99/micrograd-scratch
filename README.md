# micrograd — from scratch

A tiny autograd engine and neural network library built from scratch in Python. No PyTorch, no TensorFlow, no external ML libraries — just the `math` module and first principles.

Inspired by [Andrej Karpathy's micrograd lecture](https://www.youtube.com/watch?v=VMj-3S1tku0).

![computation graph](assets/graph.png)
*Computation graph of an MLP(2, [4, 4, 1]) forward + backward pass, visualized with graphviz*

---

## What's in here

Everything lives in a single notebook: **`neuralNetwork.ipynb`**

The notebook walks through the entire journey:

**1. The `Value` class** — a scalar value that tracks its own computation history. Supports `+`, `*`, `-`, `**`, `tanh`, and reverse-mode autodiff via `.backward()`. This is the entire autograd engine — about 60 lines.

**2. Visualization** — a `draw_dot()` function using graphviz to render the computation graph. Every node shows its data and gradient.

**3. Manual neuron test** — a single neuron computed by hand (`w1*x1 + w2*x2 + b → tanh`) with backprop and graph visualization to verify gradients.

**4. `Neuron`, `Layer`, `MLP` classes** — a minimal neural network library built on top of `Value`. Each neuron does a weighted sum + tanh activation. Layers stack neurons. MLP stacks layers.

**5. Training** — MSE loss + gradient descent on a small dataset, 100 steps. Loss goes from ~4 to ~0.

**6. XOR experiments** — the classic nonlinear problem. Compared two architectures:
  - `MLP(2, [4, 1])` — shallow
  - `MLP(2, [3, 4, 1])` — deeper, to see how additional layers help

## How it works

Every arithmetic operation on a `Value` creates a new `Value` and records the operation + parent nodes. This silently builds a directed acyclic graph (DAG). Calling `.backward()` on the output walks this graph in reverse topological order, applying the chain rule at each node to compute gradients.

```
x1 = Value(0.5)
x2 = Value(0.7)
y = model(x)       # builds the graph
y.backward()        # computes all gradients
```

That's it. That's autograd.

## Run it

```bash
pip install graphviz
jupyter notebook neuralNetwork.ipynb
```

Graphviz also needs to be installed on your system — `brew install graphviz` on Mac or `apt install graphviz` on Linux.

## What I learned

- Backpropagation is just the chain rule applied in reverse topological order
- Every `.backward()` call in PyTorch does exactly what this notebook does, just on tensors instead of scalars
- You must zero gradients before each backward pass because they accumulate with `+=`
- XOR can't be solved without a hidden layer — that's not a textbook fact anymore, I watched it fail
- Building from scratch is slow and impractical and worth every minute

## Credit

All credit to [Andrej Karpathy](https://github.com/karpathy/micrograd) for the original micrograd and the lecture that made this click.# micrograd-scratch
