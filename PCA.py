# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 14:57:34 2024

@author: mszatkow
"""

import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
from sklearn.model_selection import train_test_split

# Read face images from a folder
faces = {}

# Get the current directory path (where your script is located)
current_directory = Path(os.path.dirname(os.path.abspath(__file__)))

# Path to the folder containing the images (e.g., JPEG images)
folder_name = "BaseACPjpg"
folder_path = current_directory / folder_name  # Use Path for paths

#image hors du dataset
image_test = cv2.imread("D:/Documents/INFO5/VA51/PCA_DL_project/image_out_dataset.jpg", cv2.IMREAD_GRAYSCALE)
image_test = cv2.resize(image_test, (128, 128))

# Check if the folder exists
if folder_path.exists() and folder_path.is_dir():
    print(f"Folder path is: {folder_path}")
else:
    print(f"Folder {folder_name} does not exist in the directory {current_directory}")

# Iterate over all the files in the folder
for filename in folder_path.iterdir():
    if filename.suffix in ['.jpeg', '.jpg']:
        print("Found file:", filename.name)

        # Full path to the image file
        image_path = filename

        # Read the image in grayscale
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        # Resize the image to 128x128
        image = cv2.resize(image, (128, 128))

        # Check if the image was successfully loaded
        if image is None:
            print(f"Error loading image: {filename}")
            continue  # Skip this image if there's an error

        # Add the image to the faces dictionary
        faces[filename.name] = image

# Ensure that 'faces' contains enough images for training and testing
if len(faces) < 16:
    print("Not enough images to display a sample of 16 images.")
else:
    # Show sample faces using matplotlib
    fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(8, 10))
    faceimages = list(faces.values())[-16:]  # Take the last 16 images

    for i in range(16):
        # Ensure the image is in grayscale
        axes[i // 4][i % 4].imshow(faceimages[i], cmap="gray")
        axes[i // 4][i % 4].axis('off')  # Optionally hide the axes

    plt.subplots_adjust(wspace=0, hspace=0)  # Reduce space between subplots
    print("Showing sample faces")
    plt.show()

# Print some details
faceshape = list(faces.values())[0].shape
print("Face image shape:", faceshape)

# Prepare the dataset
facematrix = []
facelabel = []

# Flatten images and add them to facematrix, label with the filename (as a unique identifier)
for key, val in faces.items():
    facematrix.append(val.flatten())
    facelabel.append(key)

# Convert to numpy arrays
facematrix = np.array(facematrix)

# Randomly split the dataset into 70% train and 30% test
train_data, test_data, train_labels, test_labels = train_test_split(facematrix, facelabel, test_size=0.3)

test_data = np.vstack([test_data, image_test.flatten()])
print(f"Training set size: {len(train_data)}")
print(f"Test set size: {len(test_data)}")

# Apply PCA on the training set and take the first K principal components as eigenfaces
pca = PCA().fit(train_data)
n_components = 50
eigenfaces = pca.components_[:n_components]

# Show the first 16 eigenfaces
fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(8, 10))
for i in range(16):
    axes[i % 4][i // 4].imshow(eigenfaces[i].reshape(faceshape), cmap="gray")
print("Showing the eigenfaces")
plt.show()

#set timer and beginning of calculation
start_time = time.time()

# Generate weights as a KxN matrix where K is the number of eigenfaces and N is the number of training samples
train_weights = eigenfaces @ (train_data - pca.mean_).T
print("Shape of the weight matrix:", train_weights.shape)

end_time = time.time()
pca_duration = end_time - start_time

correct_matches = 0

# Test on all images in the test set
for i, query in enumerate(test_data):
    query = query.reshape(1, -1)
    query_weight = eigenfaces @ (query - pca.mean_).T
    euclidean_distance = np.linalg.norm(train_weights - query_weight, axis=0)
    best_match = np.argmin(euclidean_distance)

    #print(
       # f"Test image {i + 1} ({test_labels[i]}) best matches {train_labels[best_match]} with Euclidean distance {euclidean_distance[best_match]}")

    # Visualize the query and the best match
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 6))
    axes[0].imshow(query.reshape(faceshape), cmap="gray")
    axes[0].set_title("Query")
    axes[1].imshow(train_data[best_match].reshape(faceshape), cmap="gray")
    axes[1].set_title("Best match")
    plt.show()

    # Ask the user if the match is correct
    user_input = input("Is this match correct? (y/n): ").strip().lower()
    if user_input == 'y':
        correct_matches += 1


# Calculate and print the accuracy
accuracy = (correct_matches / len(test_data)) * 100

print(f"Accuracy: {accuracy:.2f}%")
print(f"calcul duration:{pca_duration:.3f}s")
