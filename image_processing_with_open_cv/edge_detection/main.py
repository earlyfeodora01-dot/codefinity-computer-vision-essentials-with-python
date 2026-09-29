import io
import cv2
import numpy as np
import urllib.request
from PIL import Image
import matplotlib.pyplot as plt

url = "https://content-media-cdn.codefinity.com/courses/ef049f7b-ce21-45be-a9f2-5103360b0655/task_pictures/dragon.jpeg"
with urllib.request.urlopen(url) as response:
    image_data = response.read()
image = np.array(Image.open(io.BytesIO(image_data)))

# Write your code below
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

sobel_x = cv2.Sobel(image, cv2.CV_64F,1,0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F,0,1, ksize=3)
sobel_img = cv2.magnitude(sobel_x,sobel_y)

canny_img = cv2.Canny(gray_image, 200, 300)

# Display the results
plt.figure(figsize=(10, 5))

plt.subplot(1,3,1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(sobel_img, cmap='gray')
plt.title('Sobel Edge Detection')
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(canny_img, cmap='gray')
plt.title('Canny Edge Detection')
plt.axis('off')

plt.show()


