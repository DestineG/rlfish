import numpy as np

def sample(dices=2):
    x = 0
    for _ in range(dices):
        x += np.random.randint(1, 7)
    return x

def compute_expected_value_by_general(dices=2, trials=100000):
    results = [sample(dices) for _ in range(trials)]
    V = sum(results) / trials
    print(f"Estimated expected value for rolling {dices} dice: {V}")

def compute_expected_value_by_step(dices=2, trails=100000):
    expectd, step = [0], 0
    for _ in range(trails):
        step += 1
        expectd.append(expectd[-1] + (sample(dices) - expectd[-1]) / step)
    print(f"Estimated expected value for rolling {dices} dice: {expectd[-1]}")

if __name__ == "__main__":
    compute_expected_value_by_general(dices=2, trials=100000)
    compute_expected_value_by_step(dices=2, trails=100000)