import os
import numpy as np
import cv2
from tensorflow import keras 
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set up folder and file paths
folder_path = 'E:/Anik Alvi/Intruder Detection using KOAD/Code/dataset200/'
#folder_path = 'E:/Anik Alvi/unsupervised-face-mask-detection/mtcnn-face-detection/code/mtcnn/croppedR2/'
image_files = os.listdir(folder_path)
num_images = len(image_files)

# Set up image size and flatten the images
img_size = (64, 64)
X = np.zeros((num_images, img_size[0]*img_size[1]))
for i, file in enumerate(image_files):
    img = cv2.imread(os.path.join(folder_path, file), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_size)
    X[i,:] = img.flatten()

# Normalize pixel values
X = X / 255.0

# Define the autoencoder model
input_dim = X.shape[1]
encoding_dim = 32
input_img = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(input_dim, activation='sigmoid')(encoded)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
history = autoencoder.fit(X, X, epochs=100, batch_size=64, shuffle=True)

# Get the reconstructed images
X_reconstructed = autoencoder.predict(X)

# Calculate reconstruction errors
error = np.sqrt(np.sum((X - X_reconstructed)**2, axis=1))

# Define anomaly threshold
threshold = np.percentile(error, 80)

# Define actual anomalous image indices
#actual_anomalies = [26, 49, 65, 69, 79]   
actual_anomalies = [20, 31, 52, 73, 103, 116, 126, 147, 168, 184]
#actual_anomalies = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]   
#actual_anomalies = [5, 9, 11, 12, 13, 14, 23, 24, 30, 31, 34, 38, 39, 42, 43, 46, 47, 50, 52, 53, 59, 61, 62, 64, 72]  
#actual_anomalies = [2, 3, 5, 10, 11, 12, 13, 19, 21, 22, 24, 27, 33, 36, 39, 43, 45, 57, 58, 84, 89, 98, 107, 111, 118, 120, 123, 127, 128, 140, 144, 151, 155, 166, 169, 174, 175, 177, 189, 190, 195, 200, 203] 

# Identify and print anomalous images
anomalous_indices = [i for i in range(num_images) if error[i] > threshold]
print("Anomalous image indices: ", anomalous_indices)

# Calculate detection rate and false alarm rate
num_detected = len(set(actual_anomalies).intersection(anomalous_indices))
num_false_alarms = len(anomalous_indices) - num_detected
detection_rate = (num_detected / len(actual_anomalies)) * 100
false_alarm_rate = (num_false_alarms / (num_images - len(actual_anomalies))) * 100

print("Detection rate: ", detection_rate)
print("False alarm rate: ", false_alarm_rate)

# Plot the time series of reconstruction errors
plt.stem(error)
plt.plot([0, num_images], [threshold, threshold], 'r--')
plt.xlabel('Timestep, t')
plt.ylabel('Reconstruction error')
plt.title('Dataset 2: Time series PCA anomaly detection')
plt.show()
