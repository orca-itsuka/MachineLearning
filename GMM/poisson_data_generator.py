import numpy as np
from scipy.stats import poisson


poisson_mean = 0.2
N = 10000

np.random.seed(42)
k = np.random.poisson(poisson_mean, N)

np.random.seed(42)
error = np.random.normal(0., 0.2, len(k))

noisy_data = k + error

noisy_data.reshape(10000, 1)
print(noisy_data.shape)

np.savetxt("noisy_poisson_data.csv", noisy_data, delimiter=",")
