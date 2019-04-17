'''
Example of loading genotype data from eigenstrat file, defining PCA model and estimating scores for samples with missing data.
The example data is derived from:

Human Origins data set
Genomic insights into the origin of farming in the ancient Near East
Lazaridis et.al.
Nature 2016

'''

import numpy as np
from mpca.utils.visualization import connectpoints, get_scores_by_pop, get_plot_style, plot_scores
from mpca.utils.datahandling import remove_values, read_from_EIGENSTRAT
from mpca.estimate.pcamodel import pcamodel
from mpca.estimate.lsq import projection_to_model_plane, trimmed_score_regression, known_data_regression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import seaborn as sns
sns.set()


# 249 samples, chroms 20-22
filebase = "HumanOrigins249_tiny"

# all samples, chroms 16-22
# filebase = "HumanOrigins2067_short"

# Mapping of populations to superpopulations (not part of the original data, may have mislabeled populations or other errors)
pop_superpop_file="data/HO_superpopulations"


missing_val = -1

genotypes, ind_pop_list = read_from_EIGENSTRAT("data/" +filebase + ".eigenstratgeno", "data/" + filebase + ".fam")
n_samples = len(ind_pop_list)

# set all genotypes that are originally missing to 0 for testing purposes
for r in range(genotypes.shape[0]):
	genotypes[r,:] = list(map(lambda x: 0 if x == 9.0 else x, genotypes[r,:]))


# Select how many test samples to use.
# PCA will be performed based on the train samples, and scores will be estimated for the test samples.
n_test_samples = 25
test_idx = np.random.choice(n_samples, size=n_test_samples, replace=False)
train_idx = np.arange(n_samples)
train_idx = np.delete(train_idx, test_idx)

# Centered and normalized genotype data of the train and test samples
genotypes_train = genotypes[train_idx]
genotypes_test = genotypes[test_idx]

print("train data: " + str(genotypes_train.shape))
print("test data: " + str(genotypes_test.shape))

scaler = StandardScaler()
scaler.fit(genotypes_train)
genotypes_train = scaler.transform(genotypes_train)
genotypes_test = scaler.transform(genotypes_test)


n_samples = genotypes.shape[0]
n_markers = genotypes.shape[1]


# Individual and population IDs of the train and test samples
ind_pop_list_train = ind_pop_list[train_idx]
ind_pop_list_test = ind_pop_list[test_idx]

pm = pcamodel()
pm.define_PCA_model(genotypes_train)

scores_train = pm.scores
scores_test_true = np.dot(genotypes_test, pm.loadings)

##################### plot train, test and estimated test scores using the PMP method ############

# fraction of genotypes to set to missing
missing_fraction = 0.4

genotypes_sparse = remove_values(genotypes_test, missing_fraction, missing_val)

scores_est = projection_to_model_plane(pm, genotypes_sparse, missing_val=missing_val)
mse = mean_squared_error(scores_test_true, scores_est)


# Plot the train scores
scores_train_by_pop = get_scores_by_pop(scores_train, ind_pop_list_train)
style_dict = get_plot_style(pop_superpop_file, "plots/legend_HO.png", width=1.6, height=1.9, markersize=50, fontsize=6)
plot_scores(scores_train_by_pop, style_dict, pop_superpop_file, markersize=30, figsize=(9,8))

plt.savefig("plots/pca."+filebase+".png", bbox_inches="tight")


# Plot the true and estimated test scores
plt.scatter(scores_test_true[:,0], scores_test_true[:,1], label="true test scores", color="cyan", alpha=0.8, s=50, edgecolors="black")
plt.scatter(scores_est[:,0], scores_est[:,1], label="estimated test scores", color="red", alpha=0.7, s=60, edgecolors="black")
for p in range(len(scores_test_true)):
	connectpoints(scores_test_true[p], scores_est[p])
plt.title("mse: {} ".format(mse))
plt.legend()
plt.savefig("plots/estimated_scores_"+filebase+"_"+str(n_test_samples)+"_test_"+str(missing_fraction)+".png", bbox_inches="tight")
plt.close()

##################### test the different methods on various levels of missing genotypes ############
missing_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

mses_pmp = []
mses_tri = []
mses_tsr = []
mses_kdr = []

for mf in missing_fractions:
	genotypes_sparse = remove_values(genotypes_test, mf, missing_val)
	# PMP
	scores_pmp = projection_to_model_plane(pm, genotypes_sparse, missing_val=missing_val)
	mse_pmp = mean_squared_error(scores_test_true, scores_pmp)
	mses_pmp.append(mse_pmp)

	# TSR and TRI
	scores_tsr, scores_tri = trimmed_score_regression(pm, genotypes_train, genotypes_sparse, missing_val=missing_val)
	mse_tri = mean_squared_error(scores_test_true, scores_tri)
	mse_tsr = mean_squared_error(scores_test_true, scores_tsr)
	mses_tri.append(mse_tri)
	mses_tsr.append(mse_tsr)

	# KDR
	scores_kdr = known_data_regression(pm, genotypes_train, genotypes_sparse, missing_val=missing_val)
	mse_kdr = mean_squared_error(scores_test_true, scores_kdr)
	mses_kdr.append(mse_kdr)

sns.set_style(style="whitegrid", rc=None)
fig,ax = plt.subplots(figsize=(7,6))
plt.grid(b=True, which='major', linewidth=0.5)
plt.grid(b=True, which='minor', linewidth=0.25, linestyle="--")

ax.set_yscale("log", nonposy='clip')
plt.plot(missing_fractions, mses_tri, label="TRI")
plt.plot(missing_fractions, mses_pmp, label="PMP")
plt.plot(missing_fractions, mses_tsr, label="TSR")
plt.plot(missing_fractions, mses_kdr, label="KDR")
plt.title("estimation error of PCA scores from data with missing genotypes")
plt.ylabel("log mean squared error")
plt.xlabel("fraction missing data")
plt.legend()
plt.savefig("plots/errors_"+filebase+"_"+str(n_test_samples)+"_test.png", bbox_inches="tight")
plt.show()
