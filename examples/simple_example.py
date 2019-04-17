from mpca.estimate.pcamodel import pcamodel
from mpca.estimate.lsq import projection_to_model_plane, known_data_regression
from mpca.utils.datahandling import remove_values
from mpca.utils.visualization import connectpoints
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn as sns
sns.set()
sns.set_style(style="whitegrid", rc=None)

dataset = datasets.load_wine()
n_test_samples = 10
test_idx = np.random.choice(dataset.data.shape[0], size=n_test_samples, replace=False)
train_idx = np.arange(dataset.data.shape[0])
train_idx = np.delete(train_idx, test_idx)

X = dataset.data[train_idx]
z = dataset.data[test_idx]

scaler = StandardScaler()
scaler.fit(X)
# train data used to define the PCA model
X = scaler.transform(X)
# test samples to estimate scores for
z = scaler.transform(z)

pm = pcamodel()
pm.define_PCA_model(X)

pm.save_to_h5("data/simple_model.h5")
pm.load_from_h5("data/simple_model.h5")


scores_X = pm.scores
scores_z_true = np.dot(z, pm.loadings)

# remove 25% of values from the test samples
z_sparse = remove_values(z, 0.25)
scores_pmp = projection_to_model_plane(pm, z_sparse)
scores_kdr = known_data_regression(pm, X, z_sparse)


fig,ax = plt.subplots(figsize=(7,6))
plt.scatter(scores_X[:,0], scores_X[:,1], label="train", color="gray", alpha=0.5)
plt.scatter(scores_z_true[:,0], scores_z_true[:,1], label="true", color="green", alpha=0.8)
plt.scatter(scores_pmp[:,0], scores_pmp[:,1], label="estimated PMP", color="red", alpha=0.8)
plt.scatter(scores_kdr[:,0], scores_kdr[:,1], label="estimated KDR", color="purple", alpha=0.8)

# draw lines between true score and pmp estimation
for p in range(len(scores_z_true)):
	connectpoints(scores_z_true[p], scores_pmp[p])

# draw lines between true score and kdr estimation
for p in range(len(scores_z_true)):
	connectpoints(scores_z_true[p], scores_kdr[p])

plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend()
plt.savefig("plots/wine_data.png", bbox_inches="tight")
plt.show()
plt.close()
