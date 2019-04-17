import h5py
import numpy as np

def write_h5(filename, dataname, data):
	'''
	Write data to a h5 file.
	NOTE: Replaces dataset dataname if already exists.

	:param filename: Name of the file on disk.
	:param dataname: Name the datataset.
	:param data: The data to store.
	'''
	with h5py.File(filename, 'a') as hf:
		try:
			hf.create_dataset(dataname,  data=data)
		except RuntimeError:
			print("Replacing dataset "+dataname + " in "+ filename)
			del hf[dataname]
			hf.create_dataset(dataname,  data=data)


def read_h5(filename, dataname):
	'''
	Read data from a h5 file.

	:param filename: Name of the file on disk.
	:param dataname: Name of the dataset.
	:return: The data.
	'''
	with h5py.File(filename, 'r') as hf:
		data = hf[dataname][:]
	return data

def get_pop_superpop_list(file):
	'''
	Get a list mapping populations to superpopulations from file.

	:param file: directory, filename and extension of a file mapping populations to superpopulations.
	:return: a (n_pops) x 2 list

	Assumes file contains one population and superpopulation per line, separated by ","  e.g.

	Kyrgyz,Central/South Asia

	Khomani,Sub-Saharan Africa

	'''

	pop_superpop_list = np.genfromtxt(file, usecols=(0,1), dtype=str, delimiter=",")
	return pop_superpop_list


def read_from_EIGENSTRAT(genofile, popfile):
	'''
	Read genotypes from eigenstrat file, and sample ID and population ID from another file.
	:param genofile: text file with genotypes represented as 0,1,2 (9 for missing data) in a (n_markers x n_samples) order
	:param popfile: text file containing sample IDs and their population IDs. E.g. a plink fam file,
              or a file that contains one line for each sample with the following information: "populationID sampleID"
	:return: genotypes (n_samples x n_markers)
			 ind_pop_list: array mapping individual IDs to populations so that ind_pop_list[i,0] is the individual ID
		   of sample i, and ind_pop_list[i,1] is the population of sample i, in the same order as in genotypes
	NOTE: the genotypes are transposed
	'''
	ind_pop_list = np.genfromtxt(popfile, usecols=(1,0), dtype=str)
	n_samples = len(ind_pop_list)
	# read genotypes from file
	genotypes = np.genfromtxt(genofile, delimiter = np.repeat(1, n_samples)).T
	return genotypes, ind_pop_list

def normalize_genos_EIGENSTRATstyle(genodata):
	'''
	Normalize genotypes as described in EIGENSTRAT article

	Principal components analysis corrects for stratification in genome-wide association studies
	Alkes L Price, Nick J Patterson, Robert M Plenge, Michael E Weinblatt, Nancy A Shadick & David Reich
	Nature Genetics
	2006

	Centering by mean and normalization over rows (over SNPs).

	Missing data exluded from normalization and set to value 0.

	:param genodata: genotypes represented as 0,1,2, missing values encoded as 9
	:type genodata: array, shape (n_markers x n_samples)
	:return: Centered and normalized genodata, transposed to (n_samples x n_markers).
	'''

	# mean_test = np.mean(genotypes_train, axis=0)
	# X_new = genotypes_train-mean_test
	# print(mean_test.shape)
	# print(mean_test)



	for r in range(genodata.shape[0]):
		snp_obs = 0.0
		snp_sum = 0.0

		genotypes = genodata[r,:]


		for n in range(len(genotypes)):
			if genotypes[n] < 9.0:
				snp_obs  += 1.0
				snp_sum += genotypes[n]

		snp_avg = snp_sum / snp_obs

		allele_freq_est = (snp_sum + 1.0) / ((2.0 * snp_obs) + 2.0)
		allele_freq_std  = allele_freq_est * (1.0-allele_freq_est)

		if (allele_freq_std > 0.0):
			f = 1.0 / np.sqrt(allele_freq_std)
		else:
			f = 1.0

		for n in range(len(genotypes)):
			if genotypes[n] < 9.0:
				genotypes[n] = (genotypes[n] - snp_avg) * f
			else:
				genotypes[n] = 0

		genodata[r,:] = genotypes

	return genodata.T

def remove_values(data, missing_fraction, missing_val=-1.0):
	'''
	Randomly set missing_fraction of the data to missing.

	:param data: the data
	:type data: array, shape (n_samples x n_variables)
	:param missing_fraction: fraction of data to set to missing
	:param missing_val: the value used to represent missing data
	:return:
	'''
	n_variables = data.shape[1]
	n_missing = int(n_variables*missing_fraction)
	data = np.copy(data)
	for s in range(data.shape[0]):
		missing_idx = np.random.choice(n_variables, size=n_missing, replace=False)
		data[s,missing_idx] = missing_val
	return data
