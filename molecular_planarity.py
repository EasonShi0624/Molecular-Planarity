#!/usr/bin/env python3
import numpy as np

# Use .xyz format file as input
raw_data = "molecule.xyz"

# Separete the three dimentionalities of the input coordinate
def input_processing(raw):
    data = np.loadtxt(raw, skiprows=2, usecols=(1,2,3))
    x = data[:, 0]    
    y = data[:, 1]    
    z = data[:, 2]  
    return x,y,z

# SVD and parameter generation
def fit_plane_svd(coors):
    # Get the geometric center
    centroid = np.mean(np.transpose(coors), axis=0) 
    centered_coords = np.transpose(coors) - centroid 
    # SVD
    left, mid, right = np.linalg.svd(np.transpose(centered_coords))
    mini_left_sv = left[:,-1]
    # Generate plane parameters
    u1, u2, u3 = mini_left_sv
    A, B, C = np.matrix.item(u1), np.matrix.item(u2), np.matrix.item(u3)
    #D = −(u1xc + u2yc + u3zc)
    D = - (A * np.array(centroid)[0][0] + B * np.array(centroid)[0][1] + C * np.array(centroid)[0][2])
    return A, B, C, D

# Calculate MPP and SDP
def calculate_mpp_sdp(coors, A, B, C, D,n_atom):
    # Signed distances for SDP
    signed_distances = (A * coors[:, 0] + B * coors[:, 1] + C * coors[:, 2] + D) / np.sqrt(A**2 + B**2 + C**2)
    # Distance of each atom to the plane
    distances = np.abs(signed_distances)
    # Target values
    mpp = np.sqrt((1/n_atom) * np.sum(np.array(distances) ** 2))  # Root-mean-square of deviation of the atoms from the fitting plane
    sdp = np.max(signed_distances) - np.min(signed_distances)  # Span of deviation
    return mpp, sdp, signed_distances

# Turn the coordinate file into 3*N matrix
x, y, z = np.array(input_processing(raw_data))
input_coors = np.asmatrix([x,y,z])
n_atom = len(x)

if __name__ == '__main__':
    A, B, C, D = fit_plane_svd(input_coors)
    mpp, sdp, signed_distances = calculate_mpp_sdp(np.transpose(input_coors), A, B, C, D, n_atom)
    print("MPP:", mpp)
    print("SDP:", sdp)
    print("Signed Distances:", signed_distances)
