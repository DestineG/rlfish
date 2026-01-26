import matplotlib.pyplot as plt

def init():
    return [0.0, 0.0]

def update_sync(V_k):
    res = []
    v0 = 0.5 * (-1 + 0.9 * V_k[0]) + 0.5 * (1 + 0.9 * V_k[1])
    res.append(v0)
    v1 = 0.5 * (0 + 0.9 * V_k[0]) + 0.5 * (-1 + 0.9 * V_k[1])
    res.append(v1)
    return res

def update_async(V_k):
    V_k[0] = 0.5 * (-1 + 0.9 * V_k[0]) + 0.5 * (1 + 0.9 * V_k[1])
    V_k[1] = 0.5 * (0 + 0.9 * V_k[0]) + 0.5 * (-1 + 0.9 * V_k[1])
    return V_k


if __name__ == "__main__":
    steps = 100

    # -------- 同步更新 --------
    V_k = init()
    sync_v0, sync_v1 = [], []
    for _ in range(steps):
        sync_v0.append(V_k[0])
        sync_v1.append(V_k[1])
        V_k = update_sync(V_k)

    # -------- 异步更新 --------
    V_k = init()
    async_v0, async_v1 = [], []
    for _ in range(steps):
        async_v0.append(V_k[0])
        async_v1.append(V_k[1])
        V_k = update_async(V_k)

    # -------- 画图 --------
    plt.figure(figsize=(8, 5))

    plt.plot(sync_v0, label="sync V[0]")
    plt.plot(sync_v1, label="sync V[1]")

    plt.plot(async_v0, "--", label="async V[0]")
    plt.plot(async_v1, "--", label="async V[1]")

    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title("Synchronous vs Asynchronous Bellman Updates")
    plt.legend()
    plt.grid(True)

    plt.show()
