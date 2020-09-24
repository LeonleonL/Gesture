# ----------------------------------------------------------------------------------------------------------------------
# image to video
# ----------------------------------------------------------------------------------------------------------------------

import cv2
import os

# ----------------------------------------------------------------------------------------------------------------------

writer = None
files = os.listdir('./images/')
elements = []
for file in files:
    if file.endswith('png'):
        elements.append(int(file.split('.')[0]))
elements.sort()

for file in elements:
    image_path = './images/' + str(file) + '.png'
    print(image_path)
    image = cv2.imread(image_path)

    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter("./video/video.mp4", fourcc, 20,
                                 (image.shape[1], image.shape[0]), True)

    writer.write(image)

    cv2.imshow('', image)
    if cv2.waitKey(150) & 0xFF == ord('q'):
        break

# ----------------------------------------------------------------------------------------------------------------------
