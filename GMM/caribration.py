import numpy as np
from scipy.stats import norm


scale = [0.24997013, 0.26921991, 0.38427267]
carib = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
carib0_0 = (norm.cdf(0.5, loc=0, scale=scale[0]))
            #  - norm.cdf(-0.5, loc=0, scale=scale[0]))

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
