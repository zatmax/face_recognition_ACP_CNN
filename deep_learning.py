# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 14:57:34 2024

@author: mszatkow
"""

import cv2 as cv
import numpy as np
import os
from sklearn.decomposition import PCA
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt
import time
from pathlib import Path


def GetImages(folder_path, target_size):
    """
    Fonction pour charger toutes les images dans un dossier donné,
    les redimensionner et les stocker dans un tableau NumPy.

    Parameters:
    folder_path (str): Chemin du dossier contenant les images.
    target_size (tuple): Taille cible (largeur, hauteur) pour redimensionner les images.

    Returns:
    imgs (numpy.ndarray): Tableau des images chargées et redimensionnées.
    """
    # Étape 1 : Compter le nombre d'images dans le dossier
    image_count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Filtrer les fichiers d'image
                image_count += 1

    # Étape 2 : Créer un tableau NumPy vide pour stocker les images
    imgs = np.empty((image_count, target_size[1], target_size[0]), dtype=np.uint8)  # Dimensions : (n, hauteur, largeur)

    # Étape 3 : Charger et redimensionner les images
    index = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Vérifier les fichiers d'image
                file_path = os.path.join(root, file)
                img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)  # Charger l'image en niveaux de gris
                if img is not None:
                    resized_img = cv.resize(img, target_size)  # Redimensionner l'image
                    imgs[index] = resized_img
                    index += 1

    return imgs


def GetSplitData(folder_path, image_size, test_size=0.3, normalize=True):
    """
    Charger les images et séparer les ensembles de données en formation et validation.

    Parameters:
    folder_path (str): Chemin du dossier contenant les classes d'images.
    image_size (tuple): Taille à laquelle redimensionner les images.
    test_size (float): Proportion de l'ensemble de test.
    normalize (bool): Si True, les images seront normalisées (valeurs entre 0 et 1).

    Returns:
    tuple: (X_train, X_val, y_train, y_val) - Ensembles d'entraînement et de validation.
    """
    X = []  # Liste pour stocker les images
    y = []  # Liste pour stocker les étiquettes des classes
    num_classes = 0  # Compteur pour les classes

    # Parcourir chaque dossier de classe et charger les images
    for class_name in os.listdir(folder_path):
        class_folder = os.path.join(folder_path, class_name)
        if os.path.isdir(class_folder):
            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)
                img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)  # Charger l'image en niveaux de gris
                img = cv.resize(img, image_size)  # Redimensionner l'image
                if normalize:
                    img = img / 255.0  # Normaliser l'image pour avoir des valeurs entre 0 et 1
                X.append(img)  # Ajouter l'image à X
                y.append([num_classes])  # Associer l'étiquette (classe) à l'image
        num_classes += 1

    # Convertir les listes en tableaux NumPy
    X = np.array(X)
    y = np.array(y)

    # Séparer les données en ensembles d'entraînement et de validation
    return train_test_split(X, y, test_size=test_size)


def GetClassImages(folder_path, size):
    """
    Fonction pour charger une seule image par classe, redimensionnée à la taille cible.

    Parameters:
    folder_path (str): Chemin vers le dossier contenant les classes d'images.
    size (tuple): Taille de redimensionnement des images.

    Returns:
    class_images (list): Liste des images de chaque classe.
    """
    class_images = []

    # Parcourir chaque classe et charger la première image de chaque classe
    for sub_folder_name in os.listdir(folder_path):
        sub_folder_path = os.path.join(folder_path, sub_folder_name)
        img_path = os.path.join(sub_folder_path, os.listdir(sub_folder_path)[0])  # Prendre la première image
        class_images.append(cv.resize(cv.imread(img_path, cv.IMREAD_GRAYSCALE), size))  # Redimensionner et ajouter

    return class_images


def MakeCNN(num_classes, size, X_train, X_val, y_train, y_val):
    """
    Créer un modèle de réseau de neurones convolutif (CNN).

    Parameters:
    num_classes (int): Nombre de classes.
    size (tuple): Taille des images en entrée.
    X_train, X_val, y_train, y_val: Données d'entraînement et de validation.

    Returns:
    model (keras.Model): Modèle CNN compilé.
    """
    model = models.Sequential()  # Modèle séquentiel

    # Première couche de convolution + pooling
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(size[0], size[1], 1)))
    model.add(layers.MaxPooling2D((2, 2)))

    # Deuxième couche de convolution + pooling
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Troisième couche de convolution + pooling
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Couche de flattening pour transformer les données 2D en vecteur 1D
    model.add(layers.Flatten())

    # Couche entièrement connectée (dense)
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))  # Dropout pour éviter le surapprentissage

    # Couche de sortie avec activation softmax pour la classification multi-classes
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Résumé du modèle
    model.summary()

    # Compilation du modèle avec optimiseur Adam et perte catégorielle
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


def FitCNNSplit(model, X_train, y_train, X_val, y_val, size, class_images, epochs=20):
    """
    Entraîner le modèle CNN sur l'ensemble d'entraînement et valider sur l'ensemble de validation.

    Parameters:
    model (keras.Model): Modèle CNN à entraîner.
    X_train, y_train: Données d'entraînement.
    X_val, y_val: Données de validation.
    size (tuple): Taille des images.
    class_images (list): Liste d'images de référence pour les classes.
    epochs (int): Nombre d'époques pour l'entraînement.

    Returns:
    model (keras.Model): Modèle entraîné.
    """
    # Démarrer le chronomètre pour mesurer le temps d'entraînement
    start = time.time()

    # Entraîner le modèle
    history = model.fit(X_train, y_train, epochs=epochs,
                        validation_data=(X_val, y_val))

    # Mesurer le temps d'exécution total de l'entraînement
    end = time.time()
    train_time = end - start

    # Afficher les courbes de précision pour l'entraînement et la validation
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

    # Évaluer les performances du modèle sur l'ensemble de validation
    test_loss, test_acc = model.evaluate(X_val, y_val, verbose=2)

    print(f"Temps d'entraînement : {train_time} secondes")
    print(f"Loss sur la validation: {test_loss}")
    print(f"Précision sur la validation: {test_acc}")

    # Chronométrer le processus de prédiction
    start = time.time()

    results = []

    # Boucle pour prédire toutes les images de validation
    for i in range(len(X_val)):
        prediction = model.predict(X_val[i].reshape(1, size[0], size[1], 1))
        results.append(np.argmax(prediction))

    end = time.time()
    pred_time = end - start
    print(f"Temps de prédiction : {pred_time} secondes")

    # Afficher les paires d'images (originales vs prédictions)
    num_pairs = len(X_val)
    cols = 6
    rows = (num_pairs + cols // 2 - 1) // (cols // 2)

    plt.figure(figsize=(12, rows * 3))

    for i in range(num_pairs):
        plt.subplot(rows, cols, 2 * i + 1)
        plt.xticks([]), plt.yticks([]), plt.grid(False)
        plt.imshow(X_val[i] * 255, cmap='gray')
        plt.xlabel('Original')

        plt.subplot(rows, cols, 2 * i + 2)
        plt.xticks([]), plt.yticks([]), plt.grid(False)
        plt.imshow(class_images[results[i]], cmap='gray')
        plt.xlabel('Prediction')

    plt.tight_layout()
    plt.show()

    return model


def FitCNNFull(model, X_train, y_train, X_val, y_val, epochs=20):
    """
    Réentraîner le modèle CNN sur l'ensemble complet des données d'entraînement et de validation combinées.

    Parameters:
    model (keras.Model): Modèle CNN.
    X_train, X_val, y_train, y_val: Données d'entraînement et de validation.
    epochs (int): Nombre d'époques pour l'entraînement.

    Returns:
    model (keras.Model): Modèle réentraîné.
    """
    X_all = np.concatenate((X_train, X_val), axis=0)  # Fusionner les ensembles
    y_all = np.concatenate((y_train, y_val), axis=0)

    # Entraîner le modèle sur les données complètes
    history = model.fit(X_all, y_all, epochs=epochs)

    return model


def ApplyCNN(model, img, size):
    """
    Appliquer un modèle CNN à une image pour faire une prédiction.

    Parameters:
    model (keras.Model): Modèle CNN.
    img (numpy.ndarray): Image à prédire.
    size (tuple): Taille cible pour redimensionner l'image.

    Returns:
    int: Classe prédite pour l'image.
    """
    img = img.reshape(1, size[0], size[1], 1)  # Reshaper l'image pour la prédiction
    img = img / 255  # Normaliser l'image
    prediction = model.predict(img)  # Faire la prédiction
    return np.argmax(prediction)  # Retourner la classe prédite


# Paramètres principaux
# Get the current directory path (where your script is located)
current_directory = Path(os.path.dirname(os.path.abspath(__file__)))
folder_name = "BaseACPfinale"
dataset_folder = current_directory / folder_name  # Use Path for paths
target_size = (224, 224)  # Taille des images
test_size = 0.2  # Proportion des données utilisées pour les tests
epochs = 10  # Nombre d'itérations d'entraînement pour le CNN
activate_cam = False  # Utilisation de la caméra pour prédiction en temps réel

# Entraîner le modèle CNN
class_images = GetClassImages(dataset_folder, target_size)  # Obtenir les images des classes
X_train, X_val, y_train, y_val = GetSplitData(dataset_folder, target_size, test_size)  # Charger les données
cnn = MakeCNN(14, target_size, X_train, X_val, y_train, y_val)  # Créer et compiler le modèle CNN
FitCNNSplit(cnn, X_train, y_train, X_val, y_val, target_size, class_images, epochs)  # Entraîner le modèle et valider

cv.destroyAllWindows()
