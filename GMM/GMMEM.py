import numpy as np  # numerical calculate
import matplotlib.pyplot as plt  # visualize
from collections import Counter  # count frequency
from scipy.stats import multivariate_normal  # to calculate multi norm dist
from scipy.stats import poisson  # poisson dist
# from mpl_toolkits.mplot3d import Axes3D  # plot


class GMM_EM():
    def __init__(self, K, poisson_mean):
        """constructor

        Args:
            K (int): number of cluster

        Returns:
            None.

        Note:
            eps(float): small number which prevents overflow or underflow
        """
        self.K = K
        self.eps = np.spacing(1)
        self.poisson_dist = [poisson.pmf(i, poisson_mean) for i in range(3)]

    def init_params(self, X):
        """parameter initializing method

        Args:
            X (numpy ndarray): (N,D) size of input data

        Returns:
            None.
        """
        # input sise is (N,D)
        self.N, self.D = X.shape
        # mean is generated from normal dist
        self.mu = np.array([[i for i in range(len(self.poisson_dist))]])
        self.mu = np.reshape(self.mu, [len(self.poisson_dist), self.D])
        # valance-covaliance matrix is identity
        self.sigma = np.tile(np.eye(self.D), (self.K, 1, 1))
        # weight is generated from uniform dist
        self.pi = self.poisson_dist
        # initial r will update soon
        self.r = np.random.randn(self.N, self.K)

    def gmm_pdf(self, X):
        """calculating the prob_dens_func of GMM method

        Args:
            X (numpy ndarray): (N,D) size of input data

        Returns:
            gmm_pdf (numpy ndarray): prob_dens_dist of gmm,
                                     and the size (N,K)
        """
        return np.array([self.pi[k]
                         * multivariate_normal.pdf(X, mean=self.mu[k],
                        cov=self.sigma[k]) for k in range(self.K)]).T

    def e_step(self, X):
        """calculating e_step method

        Args:
            X (numpy ndarray): (N,D) size of input data

        Note:
            bellow parameter will be updated
            self.r (numpy ndarray): (N, K)size of r
        """
        # calculate the prob_dens_func of GMM
        gmm_pdf = self.gmm_pdf(X)
        # calculate r in the log area
        log_r = (np.log(gmm_pdf)
                 - np.log(np.sum(gmm_pdf, 1, keepdims=True) + self.eps))
        # reconstruct the r from the log area
        r = np.exp(log_r)
        # prevent overflow when the time calculated r
        r[np.isnan(r)] = 1.0 / (self.K)
        # update
        self.r = r

    def m_step(self, X, params={"pi": True, "mu": True, "sigma": True}):
        """calculating m_step method

        Args:
            X (numpy ndarray): (N, D) size of input data

        Returns:
            None.

        Note:
            bellow parameter will be updated
            self.pi (numpy ndarray): (K) size of mixture rate
            self.mu (numpy ndarray): (K, D) size of mean
            self.sigma(numpy ndarray): (K, D, D) size of covaliance matrix
        """
        # calculate the parameter which maximize the Q func
        # at first, calculate N_k
        N_k = np.sum(self.r, 0)
        # optimize pi
        if params["pi"]:
            self.pi = N_k / self.N
        # optimize mu
        if params["mu"]:
            self.mu = (self.r.T @ X / (N_k[:, None] + np.spacing(1)))  # (K, D)
        # optimize sigma
        if params["sigma"]:
            r_tile = np.tile(self.r[:, :, None],
                             (1, 1, self.D)).transpose(1, 2, 0)  # (K, D, N)
            res_error = (np.tile(X[:, :, None],
                                 (1, 1, self.K)).transpose(2, 1, 0)
                         - np.tile(self.mu[:, :, None], (1, 1, self.N)))
            self.sigma = (((r_tile * res_error) @ res_error.transpose(0, 2, 1))
                          / (N_k[:, None, None] + np.spacing(1)))  # (K, D, D)

    def visualize(self, X):
        """visualize method

        Args:
            X (numpy ndarray): (N, D) size of input data

        Returns:
            None.

        Note:
            This doesn't perform plt.close(), performs only plt.show()
        """
        # execute clastering
        labels = np.argmax(self.r, 1)  # (N)
        count_freq = Counter(labels).most_common()
        label_frequency_desc = [i[0] for i in count_freq]  # (K)
        # using tab10 color map
        cm = plt.get_cmap("tab10")
        # preparing plot
        fig = plt.figure(figsize=(4, 4), dpi=300)
        ax = fig.add_subplot(111, projection="3d")
        # delete memory
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        # rotating in order to make more visible
        ax.view_init(elev=10, azim=70)
        # visualizing clusters one by one
        for k in range(len(label_frequency_desc)):
            cluster_indexes = np.where(labels == label_frequency_desc[k])[0]
            ax.plot(X[cluster_indexes, 0],
                    X[cluster_indexes, 1],
                    X[cluster_indexes, 2], "o", ms=0.5, color=cm(k))
        plt.show()

    def execute(self, X, iter_max, thr, params={"pi": True,
                                                "mu": True,
                                                "sigma": True}):
        """executing EM algorithm method

        Args:
            X (numpy ndarray): (N, D) size of input data
            iter_max (int): max update number
            thr (float): threshold of update

        Returns:
            Note.
        """
        # initialize parameters
        self.init_params(X)
        # list for record the log_lik of each iter
        log_likelyhood_list = []
        # calculate the init of log_lik
        log_likelyhood_list.append(
            np.mean(np.log(np.sum(self.gmm_pdf(X), 1) + self.eps)))
        # start updating
        for i in range(iter_max):
            # execute e_step
            self.e_step(X)
            # execute m_step
            self.m_step(X, params)
            # record this iter log_lik
            log_likelyhood_list.append(
                np.mean(np.log(np.sum(self.gmm_pdf(X), 1) + self.eps)))
            # plot the increase width of log_lik
            print("Log-likelihood: " + str(np.abs(log_likelyhood_list[i+1])))
            # if satisfy the threshold, or reach the max_iter, stop update
            if (np.abs(log_likelyhood_list[i]
                       - log_likelyhood_list[i+1])
                    < thr) or (i == iter_max - 1):
                print(f"em algorithm has stopped after {i + 1} iterations.")
                # self.visualize(X)
                print(self.pi)
                print(self.mu)
                print(self.sigma)
                break

        return self.pi, self.mu, self.sigma
