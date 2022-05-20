import pca
import numpy as np
import compress

# test PCA
X = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])

Z = pca.compute_Z(X)
COV = pca.compute_covariance_matrix(Z)
L, PCS = pca.find_pcs(COV)
Z_star = pca.project_data(Z, PCS, L, 1, 0)
print(Z_star)

# test compression  using PCA


X = compress.load_data('Data/Train/')
compress.compress_images(X,100)
