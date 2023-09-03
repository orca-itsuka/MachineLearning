from GMMEM import GMM_EM
from sample_data_loader import sample_poisson_data_loader
import numpy as np
import pandas as pd
from scipy.stats import norm


if __name__ == "__main__":
    X = sample_poisson_data_loader("noisy_poisson_data.csv")
    print(X.shape)
    model = GMM_EM(K=3, poisson_mean=0.2)
    pi_est, mu_est, sigma_est = model.execute(X, iter_max=100, thr=1e-6,
                                              params={"pi": False,
                                                      "mu": False,
                                                      "sigma": True})

    scale = [sigma_est[0, 0, 0], sigma_est[1, 0, 0], sigma_est[2, 0, 0]]
    carib = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    carib0_0 = (norm.cdf(0.5, loc=0, scale=scale[0]))
                # - norm.cdf(-0.5, loc=0, scale=scale[0]))

    carib1_0 = (norm.cdf(1.5, loc=0, scale=scale[0])
                - norm.cdf(0.5, loc=0, scale=scale[0]))

    carib2_0 = (1  # norm.cdf(2.5, loc=0, scale=scale[0])
                - norm.cdf(1.5, loc=0, scale=scale[0]))

    carib0_1 = (norm.cdf(0.5, loc=1, scale=scale[1]))
                # - norm.cdf(-0.5, loc=1, scale=scale[1]))

    carib1_1 = (norm.cdf(1.5, loc=1, scale=scale[1])
                - norm.cdf(0.5, loc=1, scale=scale[1]))

    carib2_1 = (1  # norm.cdf(2.5, loc=1, scale=scale[1])
                - norm.cdf(1.5, loc=1, scale=scale[1]))

    carib0_2 = (norm.cdf(0.5, loc=2, scale=scale[2]))
                # - norm.cdf(-0.5, loc=2, scale=scale[2]))

    carib1_2 = (norm.cdf(1.5, loc=2, scale=scale[2])
                - norm.cdf(0.5, loc=2, scale=scale[2]))

    carib2_2 = (1  # norm.cdf(2.5, loc=2, scale=scale[2])
                - norm.cdf(1.5, loc=2, scale=scale[2]))

    carib_matrix = np.array([[carib0_0, carib0_1, carib0_2],
                             [carib1_0, carib1_1, carib1_2],
                             [carib2_0, carib2_1, carib2_2]])

    inv_carib = np.linalg.inv(carib_matrix)
    print(carib_matrix)
    print(inv_carib)

    X = X.reshape(-1)
    disc_noisy_data = [round(X[j]) for j in range(len(X))]
    for i in range(len(disc_noisy_data)):
        if disc_noisy_data[i] < 0:
            disc_noisy_data[i] = 0
        elif disc_noisy_data[i] > 2:
            disc_noisy_data[i] = 2
    disc_noisy_data = np.array(disc_noisy_data)

    disc_noisy_data_series = pd.Series(disc_noisy_data)

    p_noisy = np.array([(disc_noisy_data_series == i).sum()
                        / len(X) for i in range(3)])
    p_est = np.matmul(inv_carib, p_noisy)
    print(p_est)
    print(1 - np.linalg.norm(p_est - pi_est))
