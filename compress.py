import matplotlib.pyplot as plt
import pca
import os

import numpy as np



def load_data(input_dir):
    global num_images
    global images_arr  # arr containing image data of all images, is if shape (52140,48), OG size (869,60,48)
    global r, c  # row and col of single image data
    r,c = 0,0
    num_images = 0
    images_arr = np.zeros((52140, 48))
    index = 0
    for file in os.listdir(input_dir):
        # Check whether file is a pgm
        if file.endswith(".pgm"):
            num_images += 1
            file_path = f"{input_dir}{file}"
            # call imread function
            with open(file_path, 'rb') as pgmf:
                image = plt.imread(pgmf)
                r, c = image.shape # sets the row and col for each image for later
                for pixels in image:
                    images_arr[index] = pixels
                    index += 1
    return images_arr  # need to divide by 255 to get rid of negatives?


def compress_images(DATA, k):
    Z = pca.compute_Z(DATA)

    COV = pca.compute_covariance_matrix(Z)
    L, PCS = pca.find_pcs(COV)
    Z_star = pca.project_data(Z, PCS, L, k, 0)
    output(Z_star)
    return 0


def output(Z_star):

    out_dir = "Output"
    isDir = os.path.isdir(out_dir)
    currDir = os.getcwd()
    path = os.path.join(currDir, out_dir)
    if isDir == False:
        os.mkdir(path,0o666)

    X_comp = np.reshape(Z_star,(num_images,r,c))

    for i in range(num_images):
        out_file_path = out_dir + '_' + str(i) + '.png'
        file_path = f"{path}\{out_file_path}"
        plt.imsave(file_path,X_comp[i],cmap='gray')
    return 0


