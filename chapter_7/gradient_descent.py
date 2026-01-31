import matplotlib.pyplot as plt
import torch
import numpy as np

def rosenbrock(x0, x1):
    return 100 * (x1 - x0 ** 2) ** 2 + (1 - x0) ** 2

if __name__ == "__main__":
    num_steps = 10000
    lr = 0.001
    x = torch.tensor([-1.2, 1.0], requires_grad=True)
    x_history = []
    y_history = []

    for step in range(num_steps):
        y = rosenbrock(x[0], x[1])
        y_history.append(y.item())

        x.grad = None
        y.backward()

        with torch.no_grad():
            x -= lr * x.grad

        x_history.append(x.detach().numpy().copy())

    x_history = np.array(x_history)

    # ============================
    # 绘制 x 的路径
    # ============================
    plt.figure(figsize=(6, 4))
    plt.plot(x_history[:, 0], label="x0")
    plt.plot(x_history[:, 1], label="x1")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title("Trajectory of Parameters x0 and x1")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ============================
    # 绘制 loss 曲线
    # ============================
    plt.figure(figsize=(6, 4))
    plt.plot(y_history)
    plt.xlabel("Steps")
    plt.ylabel("Rosenbrock Value")
    plt.title("Loss Decrease Over Iterations")
    plt.grid(True)
    plt.show()
