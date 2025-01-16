# resize image to 512x512

import cv2
import os

def resize_image(image_path, output_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (512, 512))
    cv2.imwrite(output_path, image)

image_path = 'dancing_real.jpg'
output_path = 'dancing_real_resized.jpg'
resize_image(image_path, output_path)