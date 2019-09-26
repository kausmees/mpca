import csv
import sys

sys.path.append("/home/krisa/mpca/")
sys.path.append("/home/krisa/mpca/utils")
sys.path.append("/home/krisa/GenoAE/")
sys.path.append("/home/krisa/GenoAE/utils")

from mpca.pcamodel import pcamodel
from mpca.estimate import trimmed_score_regression, projection_to_model_plane, known_data_regression, ind_pca_merge
from mpca.utils.datahandling import read_h5

from data_handler import data_generator_ae
# from visualization import plot_coords_by_superpop

from sklearn.metrics import mean_squared_error
import numpy as np

# model_file ="/home/kristiina/Projects/GenoAE/pca_out/data/pca.HumanOrigins249_tiny.1.smartPCAstyle.flip_False.h5"
# data = "HumanOrigins249_tiny"


# model_file ="/home/kristiina/Projects/GenoAE/pca_out/data/pca.HumanOrigins2067_short.1.smartPCAstyle.flip_False.h5"
# data = "HumanOrigins2067_short"


#local
# datadir = "/home/kristiina/Projects/GenoAE/data/"
# spopdir="/home/kristiina/Projects/GenoAE/data/"
# pcadir="/home/kristiina/Projects/GenoAE/pca_out/data/"

#UPPMAX
datadir = "/proj/snic2019-8-38/nobackup/private/ae_data/"
spopdir="/home/krisa/GenoAE/data/"
pcadir="/home/krisa/GenoAE/pca_out/data/"

#changed pyenv global from 2.7.6 to 3.6.5
# pynenv global XXX


test_id = int(sys.argv[1])
norm_mode=sys.argv[2]
data=sys.argv[3]

print("Doing data " + str(data))
print("-- test id " + str(test_id))
print("With norm mode " + norm_mode)


norm_opts = {"flip":False}
missing_val = -1.0


if data.startswith("HumanOrigins"):
	superpopulations_file = spopdir+"HO_superpopulations"
if data.startswith("1KG"):
	superpopulations_file = spopdir+"1KG_superpopulations"


model_file = pcadir+"pca."+data+"."+str(test_id)+"."+norm_mode+".flip_False.missing_val_"+str(missing_val)+".h5"
print("Tester: True coords reads from " + model_file)

# model_file ="/home/kristiina/Projects/GenoAE/pca_out/data/pca.HumanOrigins249_tiny.1.smartPCAstyle.flip_False.h5"
# data = "HumanOrigins249_tiny"

scores_true = read_h5(model_file, "scores_test")
pmodel = pcamodel()
pmodel.load_from_h5(model_file)


dg = data_generator_ae(datadir + data, test_id, normalization_mode=norm_mode, normalization_options=norm_opts, missing_val=missing_val, def_pca_scores = False)


# _, data_test, ind_pop_list_test = dg.get_test_set(0.0)
_, data_train, ind_pop_list_train, _ = dg.get_train_set(0.0)
# scores = trimmed_score_regression(pmodel, data_train, data_test, missing_val=missing_val)

# print("TRAIN DATA :" + str(data_train.shape))
# print("TRAIN DATA :" + str(np.average(data_train, axis=1).shape))

# print(np.average(data_train, axis=1))

# print("SCORES ")
# print(np.average(pmodel.scores, axis=0).shape)
# print(np.average(pmodel.scores, axis=0))


# print("LOADINGS")
# print(np.average(pmodel.loadings, axis=0).shape)
# print(np.average(pmodel.loadings, axis=0))
# print("ortho: " + str(np.dot(pmodel.loadings[:,0], pmodel.loadings[:,1] )))
# print("n0: " + str(np.linalg.norm(pmodel.loadings[:,0])))
# print("n1: " + str(np.linalg.norm(pmodel.loadings[:,1])))

# exit()
# coords_by_pop_train = get_coords_by_pop(datadir + data, pmodel.scores, ind_pop_list=ind_pop_list_train)
# plot_coords_by_superpop(coords_by_pop_train, "pca_train", superpopulations_file)
# scores_true = read_h5(model_file, "scores_test")
# coords_by_pop_test = get_coords_by_pop(datadir + data, scores_true, ind_pop_list=ind_pop_list_test)
# plot_coords_by_superpop(coords_by_pop_test, "pca_test", superpopulations_file)
#
# plt.scatter(scores_true[:,0], scores_true[:,1], color="green")
# plt.scatter(scores[:,0], scores[:,1], color="red")
# plt.show()
# plt.close()

mses_tsr = []
mses_kdr = []
mses_pmp = []
mses_tri = []
mses_procr = []


sparsifies = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]


for s in sparsifies:
	print("Starting sparsify " + str(s))

	# Data for Procrustes needs to be non-normalized ! OBS ASSUMED PCA DONE WITH STANDARD SCAALING ###
	data_train_nonnormalized, data_test_nonnormalized = dg.get_nonnormalized_data(s)
	if s == 0.9:
		print(" procr train data: ")
		print(data_train_nonnormalized[0,0:20])

		print(" procr test data: ")
		print(data_test_nonnormalized[0,0:20])


	scores_procr = ind_pca_merge(pmodel, data_train_nonnormalized, data_test_nonnormalized, missing_val=missing_val)
	mse_procr = mean_squared_error(scores_true, scores_procr)
	mses_procr.append(mse_procr)
	print("MSE Procr: " + str(mse_procr))


	d, targetdata_test, ind_pop_list_test, _ = dg.get_test_set(s)
	data_test = d[:,:,0]
	# data_test = targetdata_test
	if s == 0.0:
		assert np.allclose(targetdata_test, data_test)

	scores_pmp = projection_to_model_plane(pmodel, data_test, missing_val=missing_val)
	mse_pmp = mean_squared_error(scores_true, scores_pmp)
	print("MSE PMP: " + str(mse_pmp))
	mses_pmp.append(mse_pmp)


	scores_tsr, scores_tri = trimmed_score_regression(pmodel, data_train, data_test, missing_val=missing_val)
	mse_tsr = mean_squared_error(scores_true, scores_tsr)
	mses_tsr.append(mse_tsr)

	mse_tri = mean_squared_error(scores_true, scores_tri)
	mses_tri.append(mse_tri)
	print("MSE TSR: " + str(mse_tsr))
	print("MSE TRI: " + str(mse_tri))

	scores_kdr = known_data_regression(pmodel, data_train, data_test, missing_val=missing_val, ridge=True)
	mse_kdr = mean_squared_error(scores_true, scores_kdr)
	mses_kdr.append(mse_kdr)

	print("MSE KDR: " + str(mse_kdr))


print("KDR is Ridge")



print("TRI")
print(",".join(map(str,mses_tri)))
print("PMP")
print(",".join(map(str,mses_pmp)))
print("TSR")
print(",".join(map(str,mses_tsr)))
print("KDR")
print(",".join(map(str,mses_kdr)))
print("Procr")
print(",".join(map(str,mses_procr)))
print("________________--")

outfilename='errors_all_'+data+'_test_'+str(test_id)+"."+norm_mode+'.csv'
print("writing to " + outfilename)

with open(outfilename, mode='w') as res_file:
	res_writer = csv.writer(res_file, delimiter=',')
	res_writer.writerow(sparsifies)
	res_writer.writerow(mses_tri)
	res_writer.writerow(mses_pmp)
	res_writer.writerow(mses_tsr)
	res_writer.writerow(mses_kdr)
	res_writer.writerow(mses_procr)

