'''

Methods for estimation of scores of new samples, with missing data, from an existing PCA model.

###Methods based on least-squares regression###

* projection_to_model_plane()
* trimmed_score_regression()
* known_data_regression()

Detailed descriptions of algorithms in e.g.

Dealing with missing data in MSPC: several methods, different interpretations, some examples
Francisco Arteaga ,  Alberto Ferrer
JOURNAL OF CHEMOMETRICS
2002

Missing data methods in PCA and PLS: Score calculations with incomplete observations
Philip R.C.Nelson Paul A.Taylor John F.MacGregor
Chemometrics and Intelligent Laboratory Systems
1996


###Methods based on individual PCAs with intersecting data and subsequent merging###

* ind_pca_merge()

See e.g.

Comparing spatial maps of human population-genetic variationusing procrustes analysis
Wang et.al.
Stat Appl Genet Mol Biol
2010

Origins and genetic legacy of neolithic farmers and hunter-gatherers in Europe
Skoglund et.al.
Science
2012


for description of the method from applications in population genetics.


'''
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from mpca.pcamodel import pcamodel


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



def known_data_regression(pca_model, pca_data, new_data, missing_val=-1.0, ridge=False):
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
		if ridge:
			reg = Ridge(fit_intercept=True).fit(obs_pca_data, pca_model.scores)
		else:
			reg = LinearRegression(fit_intercept=True).fit(obs_pca_data, pca_model.scores)

		scores_est.append(reg.predict([obs_data])[0])

	return np.array(scores_est)

def _do_procrustes(target, source):
	'''
	Perform a linear transform consisting of translation, isotropic scaling and rotation/reflection on matrix source
	to minimze the squared distance to matrix target.

	X: source
	Y: target
	R: rotation/reflection
	s: scaling factor
	c: translation

	solves:
	min (R,s,c) : || (sXR' + c) - Y ||

	returns:
	(s*XR*' + c*)


	:param target: target set of points in k-space
	:type target: array, shape (n_points x k)
	:param source: set of points to transform to match target
	:type: source: array, shape (n_points x k)
	:return: transformation of source to match target

	'''
	X = np.array(source, dtype=np.double, copy=True)
	Y = np.array(target, dtype=np.double, copy=True)

	X -= np.mean(X, 0)
	Y -= np.mean(Y, 0)

	U, E, VT = np.linalg.svd(np.dot(Y.T,X))
	R = np.dot(U,VT)

	scale = E.sum()/np.trace(np.dot(X.T,X))

	transformed = np.dot(X, R.T) * scale
	translation = np.mean(target, 0) - np.mean(transformed, 0)
	transformed += translation

	return transformed


def ind_pca_merge(pca_model, pca_data, new_data, missing_val=-1.0, merge="procrustes"):
	'''
	Estimate scores for the given new samples (that may have missing data) for an existing PCA model
	by perfoming individual PCAs using the values of the original PCA data that overlap with observed values of the new
	samples, and subsequently merging them by transforming the scores of the individual PCA to those of the full PCA.


	:param pca_model: The PCA model to use for estimating scores.
	:type pca_model: pcamodel
	:param pca_data: The data to use to define the PCA model. Assumed NOT normalized.
	:type pca_data: array, shape (n_pca_samples x n_features)
	:param new_data: Samples to estimate scores for, based on the PCA model. Assumed NOT normalized, and that the features are
					 the same as those of pca_data, in the same order.
	:type new_data: array, shape (n_new_samples x n_features)
	:param missing_val: Value used to represent missing data.
	:param merge: procrustes: merge using procrustes transformation
				  lsq: merge using general affine transformation
	:type merge: str
	:return: scores_est : array, shape (n_new_samples x 2) Estimated scores of the new data.
	'''

	assert merge == "procrustes" or merge == "lsq", str(merge) + " is not a valid merge method"

	n_new_samples = new_data.shape[0]
	scores_est = []
	scaler = StandardScaler()

	for s in range(n_new_samples):
		this_sample = new_data[s]
		obs_idx = np.where(this_sample != missing_val)[0]
		obs_data = new_data[s][obs_idx]
		obs_pca_data = pca_data[:,obs_idx]

		scaler.fit(obs_pca_data)
		obs_pca_data = scaler.transform(obs_pca_data)

		obs_data = scaler.transform([obs_data])

		pm = pcamodel()
		pm.define_PCA_model(obs_pca_data)

		scores_sparse = np.append(pm.scores,np.dot(obs_data, pm.loadings), axis=0)
		scores_full = np.append(pca_model.scores, [[0,0]], axis=0)
		if merge == "procrustes":
			scores_sparse_transformed = _do_procrustes(scores_full, scores_sparse)
			scores = scores_sparse_transformed[-1]


		elif merge == "lsq":
			reg = LinearRegression(fit_intercept=True).fit(scores_sparse, scores_full)
			scores = reg.predict([scores_sparse[-1]])[0]


		scores_est.append(scores)

	return np.array(scores_est)
