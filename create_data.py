# ----------------------------------------------------------------------------------------------------------------------
# Create data
# ----------------------------------------------------------------------------------------------------------------------

from data.hand_tracker import HandTracker
import imutils
import cv2
import csv

f = open('./data/out.csv', 'a', encoding='utf-8')

csv_writer = csv.writer(f)

connections = [# csv_writer.writerow(["class",
#                      "x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "x5", "y5",
#                      "x6", "y6", "x7", "y7", "x8", "y8", "x9", "y9", "x10", "y10", "x11", "y11",
#                      "x12", "y12", "x13", "y13", "x14", "y14", "x15", "y15", "x16", "y16", "x17", "y17",
#                      "x18", "y18", "x19", "y19", "x20", "y20"])
    (0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12), (13, 14),
    (14, 15), (15, 16), (17, 18), (18, 19), (19, 20), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
]

# ----------------------------------------------------------------------------------------------------------------------

PALM_MODEL_PATH = "./data/palm_detection_without_custom_op.tflite"
LANDMARK_MODEL_PATH = "./data/hand_landmark.tflite"
ANCHORS_PATH = "./data/anchors.csv"

detector = HandTracker(
    PALM_MODEL_PATH,
    LANDMARK_MODEL_PATH,
    ANCHORS_PATH,
    box_shift=0.2,
    box_enlarge=1.3
)

# ----------------------------------------------------------------------------------------------------------------------

POINT_COLOR = (0, 255, 0)
CONNECTION_COLOR = (255, 0, 0)
THICKNESS = 2
cap = cv2.VideoCapture(0)

# ----------------------------------------------------------------------------------------------------------------------

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=720)
    (H, W) = frame.shape[:2]
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    points, _ = detector(image)

    if points is not None:
        l = []
        for point in points:
            x, y = point
            cv2.circle(frame, (int(x), int(y)), THICKNESS * 2, POINT_COLOR, THICKNESS)
            l.append(str(x / W)[:4])
            l.append(str(y / H)[:4])

        if len(l) > 1:
            print(len(l))
            csv_writer.writerow(['4',
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

    cv2.imshow("", frame)
    key = cv2.waitKey(20)
    if key == 27:
        f.close()
        break

f.close()

# ----------------------------------------------------------------------------------------------------------------------
