import cv2
import numpy as np
from matplotlib import pyplot as plt

col_img = cv2.cvtColor(cv2.imread("komodo_dragon.jpg"),  cv2.COLOR_RGB2BGR)
fig = plt.figure(figsize=(10, 3))
rows = 1
columns = 4
fig.add_subplot(rows, columns, 1) 
plt.imshow(col_img)
plt.axis('off') 
plt.title("Original")

gray_img = cv2.cvtColor(col_img, cv2.COLOR_BGR2GRAY)
fig.add_subplot(rows, columns, 2)
plt.imshow(gray_img, cmap='gray')
plt.axis('off') 
plt.title("Grayscale")

hist_equalized = cv2.equalizeHist(gray_img)
fig.add_subplot(rows, columns, 3) 
plt.imshow(hist_equalized, cmap='gray')
plt.axis('off') 
plt.title("Histogram Equalized")

_, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=1)
fig.add_subplot(rows, columns, 4) 
plt.imshow(opening, cmap='gray')
plt.axis('off') 
plt.title("Morphological Opening")

plt.savefig("Image_Engancement_output.png")
