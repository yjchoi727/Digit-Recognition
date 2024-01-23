import cv2
import numpy as np
import math
import threading
from collections import deque
from keras.models import load_model
from objloader_simple import OBJ, ARObject

class VideoCapture:
    def __init__(self, name, res=(320, 240)):
        self.cap = cv2.VideoCapture(name)
        self.cap.set(3, res[0])
        self.cap.set(4, res[1])
        self.q = deque()
        self.status = "init"

        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

        while self.status == "init":
            pass

        assert self.status == "capture", "Failed to open capture"

    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[error] ret")
                break

            self.q.append(frame)

            self.status = "capture"

            while len(self.q) > 1:
                self.q.popleft()

        self.status = "failed"

    def read(self):
        return self.q[-1]

    def release(self):
        self.cap.release()

def get_numbers(y_pred):
    for number, per in enumerate(y_pred[0]):
        if per != 0:
            final_number = str(int(number))
            per = round((per * 100), 2)
            return final_number, per
    return None

def projection_matrix(camera_parameters, homography):
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]

    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l

    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)

    projection = np.stack((rot_1, rot_2, rot_3, translation)).T

    return np.dot(camera_parameters, projection)

def render(frame, obj, projection, referenceImage, scale3d, color=False):
    vertices = obj.obj.vertices
    scale_matrix = np.eye(3) * obj.scale
    h, w = referenceImage.shape

    for face in obj.obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)

        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        framePts = np.int32(dst)

        cv2.fillConvexPoly(frame, framePts, (137, 27, 211))

    return frame

def main():
    obj_0 = ARObject("./models/obj_0.obj", scale=0.1)
    obj_1 = ARObject("./models/obj_1.obj", scale=0.1)

    scale3d = 8
    camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
    MIN_MATCHES = 15

    referenceImage = cv2.imread("./img/referenceImage.jpg", 0)
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    referenceImagePts, referenceImageDsc = orb.detectAndCompute(referenceImage, None)

    cap = VideoCapture(0)

    while True:
        frame = cap.read()

        sourceImagePts, sourceImageDsc = orb.detectAndCompute(frame, None)

        matches = bf.match(referenceImageDsc, sourceImageDsc)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) > MIN_MATCHES:
            sourcePoints = np.float32([referenceImagePts[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            destinationPoints = np.float32([sourceImagePts[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            homography, _ = cv2.findHomography(sourcePoints, destinationPoints, cv2.RANSAC, 5.0)

            h, w = referenceImage.shape
            corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            transformedCorners = cv2.perspectiveTransform(corners, homography)

            frame = cv2.polylines(frame, [np.int32(transformedCorners)], True, 255, 3, cv2.LINE_AA)

            result = get_numbers(y_pred)
            if result is not None:
                num, per = result
                if num == '0':
                    frame = render(frame, obj_0, projection, referenceImage, scale3d, False)
                elif num == '1':
                    frame = render(frame, obj_1, projection, referenceImage, scale3d, False)

            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            print("Not enough matches are found - %d/%d" % (len(matches), MIN_MATCHES))

    cap.release()
    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    main()
