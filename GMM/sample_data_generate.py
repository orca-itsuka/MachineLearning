import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm


N1 = 4000
N2 = 3000
N3 = 2000
N4 = 1000

Mu1 = [5, -5, -5]
Mu2 = [-5, 5, 5]
Mu3 = [-5, -5, -5]
Mu4 = [5, 5, 5]

Sigma1 = [[1, 0, -0.25], [0, 1, 0], [-0.25, 0, 1]]
Sigma2 = [[1, 0, 0], [0, 1, -0.25], [0, -0.25, 1]]
Sigma3 = [[1, 0.25, 0], [0.25, 1, 0], [0, 0, 1]]
Sigma4 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

X1 = np.random.multivariate_normal(Mu1, Sigma1, N1)
X2 = np.random.multivariate_normal(Mu2, Sigma2, N2)
X3 = np.random.multivariate_normal(Mu3, Sigma3, N3)
X4 = np.random.multivariate_normal(Mu4, Sigma4, N4)

fig = plt.figure(figsize=(4, 4), dpi=300)
ax = fig.add_subplot(111, projection='3d')

cm = plt.get_cmap("tab10")

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.view_init(elev=10, azim=70)

ax.plot(X1[:, 0], X1[:, 1], X1[:, 2], "o", ms=0.5, color=cm(0))
ax.plot(X2[:, 0], X2[:, 1], X2[:, 2], "o", ms=0.5, color=cm(1))
ax.plot(X3[:, 0], X3[:, 1], X3[:, 2], "o", ms=0.5, color=cm(2))
ax.plot(X4[:, 0], X4[:, 1], X4[:, 2], "o", ms=0.5, color=cm(3))
plt.show()

# concatenate 4 classes
X = np.concatenate([X1, X2, X3, X4])

# preparing plot
fig = plt.figure(figsize=(4, 4), dpi=300)
ax = fig.add_subplot(111, projection="3d")

# delete memory
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# rotate in order to make more visible
ax.view_init(elev=10, azim=70)

# plot
ax.plot(X[:, 0], X[:, 1], X[:, 2], "o", ms=0.5, color=cm(0))
plt.show()

# save this sample data into csv
np.savetxt("data.csv", X, delimiter=",")
