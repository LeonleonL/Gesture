# ----------------------------------------------------------------------------------------------------------------------
# rotate
# ----------------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import random
import math
import csv
import cv2

# ----------------------------------------------------------------------------------------------------------------------

TRAINING_DATASET = "./data/train.csv"
df = pd.read_csv(TRAINING_DATASET, index_col=False)

X = np.array(df.drop(['class'], 1))
Y = np.array(df['class'])

H, W, C = 405, 710, 3


# ----------------------------------------------------------------------------------------------------------------------

def rotate(angle, valuex, valuey, pointx, pointy):
    valuex = np.array(valuex)
    valuey = np.array(valuey)
    nRotatex = (valuex - pointx) * math.cos(angle) - (valuey - pointy) * math.sin(angle) + pointx
    nRotatey = (valuex - pointx) * math.sin(angle) + (valuey - pointy) * math.cos(angle) + pointy
    return nRotatex, nRotatey

# ----------------------------------------------------------------------------------------------------------------------


f = open('./data/out.csv', 'a', encoding='utf-8')
csv_writer = csv.writer(f)
csv_writer.writerow(["class",
                     "x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "x5", "y5",
                     "x6", "y6", "x7", "y7", "x8", "y8", "x9", "y9", "x10", "y10", "x11", "y11",
                     "x12", "y12", "x13", "y13", "x14", "y14", "x15", "y15", "x16", "y16", "x17", "y17",
                     "x18", "y18", "x19", "y19", "x20", "y20"])

for idx in range(0, len(df)):
    for x in range(0, 50):
        img = np.zeros((H, W, C), np.uint8)
        img.fill(0)
        angle = 18 * random.randint(-10, 10)

        point_x = int(X[idx][0] * W)
        point_y = int(X[idx][1] * H)

        count = 0
        points = []
        l = []

        if 0 < point_x < W and 0 < point_y < H:
            l.append(str(point_x / W)[:4])
            l.append(str(point_y / H)[:4])
            count += 1

        for i in range(1, 21):
            x = int(X[idx][2*i] * W)
            y = int(X[idx][2*i + 1] * H)
            sPointx, sPointy = rotate(math.radians(angle), x, y, point_x, point_y)

            if 0 < sPointx < W and 0 < sPointy < H:
                count += 1
                points.append((sPointx, sPointy))

                l.append(str(sPointx / W)[:4])
                l.append(str(sPointy / H)[:4])

        if count == 21:
            for point in points:
                cv2.circle(img, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)

            print(l)
            print(len(l))
            csv_writer.writerow([Y[idx],
                                 l[0], l[1], l[2], l[3],
                                 l[4], l[5], l[6], l[7],
                                 l[8], l[9], l[10], l[11],
                                 l[12], l[13], l[14], l[15],
                                 l[16], l[17], l[18], l[19],
                                 l[20], l[21], l[22], l[23],
                                 l[24], l[25], l[26], l[27],
                                 l[28], l[29], l[30], l[31],
                                 l[32], l[33], l[34], l[35],
                                 l[36], l[37], l[38], l[39],
                                 l[40], l[41]])
            # cv2.imshow('', img)
            # cv2.waitKey(0)

f.close()