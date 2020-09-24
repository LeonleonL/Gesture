# ----------------------------------------------------------------------------------------------------------------------
# KNN predict
# ----------------------------------------------------------------------------------------------------------------------

from data.hand_tracker import HandTracker
import imutils
import cv2
import pickle
import os


MODEL_BINARY_FILE = 'knnclassifier_file'
model = pickle.load(open(os.path.abspath(f"./model/{MODEL_BINARY_FILE}"), "rb"))

# ----------------------------------------------------------------------------------------------------------------------
#        8   12  16  20
#        |   |   |   |
#        7   11  15  19
#    4   |   |   |   |
#    |   6   10  14  18
#    3   |   |   |   |
#    |   5---9---13--17
#    2    \         /
#     \    \       /
#      1    \     /
#       \    \   /
#        ------0-

connections = [
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

    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=720)
    (H, W) = frame.shape[:2]
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    points, _ = detector(image)

    l = []
    if points is not None:
        for point in points:
            x, y = point
            cv2.circle(frame, (int(x), int(y)), THICKNESS * 2, POINT_COLOR, THICKNESS)
            l.append(str(x / W)[:4])
            l.append(str(y / H)[:4])

        for connection in connections:
            x0, y0 = points[connection[0]]
            x1, y1 = points[connection[1]]
            cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), CONNECTION_COLOR, THICKNESS)


    if len(l) > 0:
        out = model.predict([l])
        cv2.putText(frame, 'predict: ' + str(out[0]), (20, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)

    cv2.imshow("", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ----------------------------------------------------------------------------------------------------------------------
