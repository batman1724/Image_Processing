import cv2

img = cv2.imread('input.jpeg')

# Extract the RGB channels from the image
r, g, b = cv2.split(img)

# Write the image dimensions to the beginning of the file
with open('input.txt', 'w') as f:
    f.write('{} {}\n'.format(img.shape[1], img.shape[0]))

# Append the pixel values to the file
with open('input.txt', 'a') as f:
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            f.write('{} {} {}\n'.format(r[i, j], g[i, j], b[i, j]))