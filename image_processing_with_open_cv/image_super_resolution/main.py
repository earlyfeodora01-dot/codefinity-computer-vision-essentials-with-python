import io
import cv2
import numpy as np
import urllib.request
from PIL import Image
import matplotlib.pyplot as plt

url_img = "https://staging-content-media-cdn.codefinity.com/courses/ef049f7b-ce21-45be-a9f2-5103360b0655/task_pictures/low_res_puppy.jpg"
with urllib.request.urlopen(url_img) as img_resp:
    image_data = img_resp.read()

url_model = "https://staging-content-media-cdn.codefinity.com/courses/ef049f7b-ce21-45be-a9f2-5103360b0655/functions/ESPCN_x4.pb"
model_path = "ESPCN_x4.pb"
with urllib.request.urlopen(url_model) as model_resp, open(model_path, "wb") as model_file:
    model_file.write(model_resp.read())

image = np.array(Image.open(io.BytesIO(image_data)))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Write your code below
bicubic_image = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(model_path)
sr.setModel("espcn", 4)

dnn_image = sr.upsample(image)

# Display the results
plt.figure(figsize=(15, 7.5))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Low-Resolution Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(bicubic_image, cv2.COLOR_BGR2RGB))
plt.title("Super-Resolved Image (Bicubic)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(dnn_image, cv2.COLOR_BGR2RGB))
plt.title("Super-Resolved Image (ESPCN)")
plt.axis("off")

plt.show()
