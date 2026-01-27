import numpy as np

x = np.array([1, 2, 3])
pi = np.array([0.1, 0.1, 0.8])

e = np.sum(x * pi)
std = np.sqrt(np.sum((x - e) ** 2 * pi))
print(e, std)

# MC
N = 100000
samples = np.random.choice(x, size=N, p=pi)
e_mc = np.mean(samples)
std_mc = np.std(samples) / np.sqrt(N)
print(e_mc, std_mc)

# Importance Sampling
b = np.array([0.3, 0.4, 0.3])
samples_is = np.random.choice(x, size=N, p=b)
w = pi[samples_is - 1] / b[samples_is - 1]
e_is = np.mean(samples_is * w)
std = np.std(samples_is * w) / np.sqrt(N)
print(e_is, std)
