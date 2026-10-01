import io
import cv2
import numpy as np
import urllib.request
from PIL import Image
import matplotlib.pyplot as plt

url_factory = "https://content-media-cdn.codefinity.com/courses/ef049f7b-ce21-45be-a9f2-5103360b0655/task_pictures/factory.jpg"
with urllib.request.urlopen(url_factory) as response:
    image_factory_data = response.read()
factory = np.array(Image.open(io.BytesIO(image_factory_data)))

url_sunflower = "https://content-media-cdn.codefinity.com/courses/ef049f7b-ce21-45be-a9f2-5103360b0655/task_pictures/sunflowers.jpg"
with urllib.request.urlopen(url_sunflower) as response:
    image_sunflower_data = response.read()
sunflowers = np.array(Image.open(io.BytesIO(image_sunflower_data)))

# Write your code below
gray_factory = cv2.cvtColor(factory, cv2.COLOR_BGR2GRAY)
gray_sunflowers =cv2.cvtColor(sunflowers, cv2.COLOR_BGR2GRAY)

# Harris Corner Detection
gray_float = np.float32(gray_factory)

harris_corners = cv2.cornerHarris(gray_factory, blockSize=2, ksize=3, k=0.04)
harris_corners = cv2.dilate(harris_corners, None)

# Shi-Tomasi Corner Detection
shi_tomasi_corners = cv2.goodFeaturesToTrack(gray_factory, maxCorners=100, qualityLevel= 0.01, minDistance=10)

# Blob Detection
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.maxThreshold = 127  
# Filter by Area
params.filterByArea = True
params.minArea = 100   # Lower to detect smaller objects
params.maxArea = 500  # Allow larger blobs
# Filter by Circularity (Allow more irregular shapes)
params.filterByCircularity = True
params.minCircularity = 0.02  
# Filter by Convexity (Allow concave blobs)
params.filterByConvexity = True
params.minConvexity = 0.5  
# Filter by Inertia (Detect elongated shapes)
params.filterByInertia = True
params.minInertiaRatio = 0.001 

detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(gray_factory)

# Mark detected corners in blue
image_harris = factory.copy()
image_harris[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255]

# Draw detected corners in green
image_shi_tomasi = factory.copy()
if shi_tomasi_corners is not None:
    for corner in shi_tomasi_corners:
        x, y = corner[0].astype(int)
        cv2.circle(image_shi_tomasi, (x, y), 5, (0, 255, 0), -1)

# Draw detected blobs in pink
image_blobs = cv2.drawKeypoints(sunflowers, keypoints, None, (255, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display the results
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_harris)
plt.title('Harris Corner Detection')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(image_shi_tomasi)
plt.title('Shi-Tomasi Corner Detection')
plt.axis('off')

plt.show()

plt.imshow(image_blobs)
plt.title('Blob Detection')
plt.axis('off')

plt.show()
