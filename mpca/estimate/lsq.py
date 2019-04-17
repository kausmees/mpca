'''

Least-squares based methods for estimation of scores of new samples, with missing data, from an existing PCA model.
Detailed descriptions of algorithms in e.g.

Dealing with missing data in MSPC: several methods, different interpretations, some examples
Francisco Arteaga ,  Alberto Ferrer
JOURNAL OF CHEMOMETRICS
2002

Missing data methods in PCA and PLS: Score calculations with incomplete observations
Philip R.C.Nelson Paul A.Taylor John F.MacGregor
Chemometrics and Intelligent Laboratory Systems
1996

'''
import numpy as np
from sklearn.linear_model import LinearRegression

def projection_to_model_plane(pca_model, new_data, missing_val=-1.0):
	'''
	Estimate scores for the given samples (that may have missing data) based on existing PCA model, using the
	Projection to the Model Plane (PMP) method.

	:param pca_model: The PCA model to use for estimating scores.
	:type pca_model: pcamodel
	:param pca_data: The data that was used to define the PCA model. Assumed normalized, and that samples and features are
					 in the same order as pca_model.scores and pca_model.loadings.
	:type pca_data: array, shape (n_pca_samples x n_features)
	:param new_data: Samples to estimate scores for, based on the PCA model. Assumed normalized, and that the features are
					 the same as those of pca_data, in the same order.
	:type new_data: array, shape (n_new_samples x n_features)
	:param missing_val: Value used to represent missing data.
	:return: scores_est : array, shape (n_new_samples x 2) Estimated scores of the new data.
	'''

	n_new_samples = new_data.shape[0]
	scores_est = []
	for s in range(n_new_samples):
		this_sample = new_data[s]
		obs_idx = np.where(this_sample != missing_val)[0]
		obs_loadings = pca_model.loadings[obs_idx]
		obs_data = new_data[s][obs_idx]
		reg = LinearRegression(fit_intercept=False).fit(obs_loadings, obs_data)
		scores_est.append(reg.coef_.T)

	return np.array(scores_est)



def trimmed_score_regression(pca_model, pca_data, new_data, missing_val=-1.0):
	'''
	Estimate scores for the given samples (that may have missing data) based on existing PCA model, using the
	Trimmed Score Regresison (TSR) method.

	Also returns the trimmed scores (estimated scores using the Trimmed Score method (TRI)).

	:param pca_model: The PCA model to use for estimating scores.
	:type pca_model: pcamodel
	:param pca_data: The data that was used to define the PCA model. Assumed normalized, and that samples and features are
					 in the same order as pca_model.scores and pca_model.loadings.
	:type pca_data: array, shape (n_pca_samples x n_features)
	:param new_data: Samples to estimate scores for, based on the PCA model. Assumed normalized, and that the features are
					 the same as those of pca_data, in the same order.
	:type new_data: array, shape (n_new_samples x n_features)
	:param missing_val: Value used to represent missing data.
	:return: scores_est_tsr : array, shape (n_new_samples x 2) Estimated scores of the new data using TSR.
		 	 scores_est_tri : array, shape (n_new_samples x 2) Estimated scores of the new data using TRI.
	'''

	n_new_samples = new_data.shape[0]
	scores_est = []
	tri_scores_est=[]

	for s in range(n_new_samples):
		this_sample = new_data[s]
		obs_idx = np.where(this_sample != missing_val)[0]
		obs_loadings = pca_model.loadings[obs_idx]
		obs_data = new_data[s][obs_idx]
		trimmed_scores = np.matmul(obs_data, obs_loadings)
		tri_scores_est.append(trimmed_scores)
		obs_pca_data = pca_data[:,obs_idx]
		trimmed_scores_pca = np.matmul(obs_pca_data, obs_loadings)
		reg = LinearRegression(fit_intercept=True).fit(trimmed_scores_pca, pca_model.scores)
		scores_est.append(reg.predict([trimmed_scores])[0])

	return np.array(scores_est), np.array(tri_scores_est)



def known_data_regression(pca_model, pca_data, new_data, missing_val=-1.0):
	'''
	Estimate scores for the given samples (that may have missing data) based on existing PCA model, using the
	Known Data Regression (KDR) method.

	:param pca_model: The PCA model to use for estimating scores.
	:type pca_model: pcamodel
	:param pca_data: The data that was used to define the PCA model. Assumed normalized, and that samples and features are
					 in the same order as pca_model.scores and pca_model.loadings.
	:type pca_data: array, shape (n_pca_samples x n_features)
	:param new_data: Samples to estimate scores for, based on the PCA model. Assumed normalized, and that the features are
					 the same as those of pca_data, in the same order.
	:type new_data: array, shape (n_new_samples x n_features)
	:param missing_val: Value used to represent missing data.
	:return: scores_est : array, shape (n_new_samples x 2) Estimated scores of the new data.
	'''


	n_new_samples = new_data.shape[0]
	scores_est = []

	for s in range(n_new_samples):
		this_sample = new_data[s]
		obs_idx = np.where(this_sample != missing_val)[0]
		obs_data = new_data[s][obs_idx]
		obs_pca_data = pca_data[:,obs_idx]
		reg = LinearRegression(fit_intercept=False).fit(obs_pca_data, pca_model.scores)
		scores_est.append(reg.predict([obs_data])[0])

	return np.array(scores_est)
