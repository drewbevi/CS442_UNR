import numpy as np

def compute_Z(X, centering=True, scaling=False):
    Z = X  # Copy of sample
    if centering:  # if true then the mean is subtracted from each feature
        Z = Z - np.mean(Z, axis=0)
    if scaling:  # if true then each feature is divided by the SD
        Z /= np.std(Z, axis=0)
    return Z


def compute_covariance_matrix(Z):
    COV = np.cov(Z, rowvar=False)
    return COV


def find_pcs(COV):
    eigen_values, eigen_vectors = np.linalg.eigh(COV)
    arr_index = np.argsort(eigen_values)[::-1]

    sorted_eigenvalues = eigen_values[arr_index]
    sorted_eigenvectors = eigen_vectors[:, arr_index]
    return sorted_eigenvalues, sorted_eigenvectors


def project_data(Z, PCS, L, k, var): # k = 1, var = 0
    if var == 0:
        e_subset = PCS[:, 0:k]
        Z_star = np.dot(e_subset.transpose(), Z.transpose()).transpose()
        return Z_star
    elif k == 0:
        e_subset = PCS[:, 0:var]
        Z_star = np.dot(e_subset.transpose(), Z.transpose()).transpose()
        return Z_star

