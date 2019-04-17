from ..utils.datahandling import read_h5, write_h5
from sklearn.decomposition import PCA
import numpy as np


class pcamodel:
	'''
	Wrapper class for a 2-component PCA model.

	PCA is performed using sklearn PCA.

	The model is defined by the loadings and the scores.

	Implements saving and loading from file.

	'''
	def define_PCA_model(self, X):
		'''
		Define a PCA model based on the data X.

		:param X: The data to define the PCA model on.
		:type X: array, shape (n_samles x n_features)

		NOTE: X must be centered
			  since pmodel uses the scikit PCA model which automatically centers the input
		  	  and the loadings need to be defined so that multiplication of new data with them is enough for a correct transform
		'''
		pca = PCA(n_components=2)
		means = np.mean(X, axis=0)
		assert np.allclose(means, np.zeros(means.shape)) , "Input data is not centered"
		pca.fit(X)
		self.scores = pca.transform(X)
		self.loadings = pca.components_.T


	def load_from_h5(self, filename):
		'''
		Load scores and loadings from a h5 file.

		:param filename: Full filename to load from.
		'''
		self.loadings = read_h5(filename, "loadings")
		self.scores = read_h5(filename, "scores")


	def load_from_SMARTPCA(self, fileprefix):
		'''
		Load scores and loadings from SMARTPCA output.

		:param fileprefix: Path and filename (exluding ending) of files to read from.

		Assumes fileprefix.evec and fileprefix.weights exists.

		'''
		raise NotImplementedError



	def save_to_h5(self, filename):
		'''
		Save the PCA model (scores and loadings) to a h5 file.

		:param filename: full filename to write to
		'''
		write_h5(filename, "loadings", self.loadings)
		write_h5(filename, "scores", self.scores)
