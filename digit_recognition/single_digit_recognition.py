import math

import cv2
import numpy as np
from keras.models import load_model

from objloader_simple import OBJ

model = load_model('C:/Users/727ke/Desktop/VRAR2/Marker_Based_AR/digit_recognition/models/tf-cnn-model.h5')

# 다른 숫자에 대한 OBJ 파일 로드
obj_files = [f"C:/Users/727ke/Desktop/VRAR2/Marker_Based_AR/digit_recognition/models/obj_{i}.obj" for i in range(10)]
objs = [OBJ(obj_file, swapyz=True) for obj_file in obj_files]

def get_numbers(y_pred):
    results = []
    for number, per in enumerate(y_pred[0]):
        if per != 0:
            final_number = str(int(number))
            per = round((per * 100), 2)
            results.append((final_number, per))
    return results

video = cv2.VideoCapture(0)


def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]

    # Normalize vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l

    # Compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(
        c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2)
    )
    rot_2 = np.dot(
        c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2)
    )
    rot_3 = np.cross(rot_1, rot_2)

    # Compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T

    return np.dot(camera_parameters, projection)

def render(frame, obj, bounding_box, projection):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 8

    # Scale the 3D model
    vertices = np.dot(vertices, scale_matrix)

    # Translate the 3D model based on the bounding box
    x, y, box_w, box_h = bounding_box
    vertices[:, 0] += x + box_w / 2
    vertices[:, 1] += y + box_h / 2

    vertices[:, 1] = 2 * bounding_box[1] + bounding_box[3] - vertices[:, 1]

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        frame_pts = np.int32(dst)

        # Fill the convex polygon on the frame
        cv2.fillConvexPoly(frame, frame_pts, color=(137, 27, 211))

    return frame


if video.isOpened():
    while True:
        check, img = video.read()
        img2 = img.copy()
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_gau = cv2.GaussianBlur(img_gray, (5, 5), 0)
        ret, thresh = cv2.threshold(img_gau, 120, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(thresh, kernel, iterations=1)

        edged = cv2.Canny(dilation, 50, 250)

        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            for contour in contours:
                if cv2.contourArea(contour) > 1000:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(img2, (x, y), (x + w, y + h), (255, 0, 255), 3)

                    new_img = thresh[y:y + h, x:x + w]
                    new_img2 = cv2.resize(new_img, (28, 28))
                    im2arr = np.array(new_img2)
                    im2arr = im2arr.reshape(1, 28, 28, 1)
                    y_pred = model.predict(im2arr)

                    results = get_numbers(y_pred)
                 
                    for result in results:
                        num, per = result
                        num_str = str(num)
                        y_p = f'{num_str}'
                        cv2.putText(img2, y_p, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                        # Calculate projection matrix (example, you should replace it with your actual calculation)
                        homography = np.eye(3)  
                        camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
                        projection = projection_matrix(camera_parameters, homography)

                        # Render .obj file based on the detected number
                        obj_index = int(num)
                        render_frame = render(img2, objs[obj_index], (x, y, w, h), projection)

        cv2.imshow("Frame", img2)
        cv2.imshow("Contours Frame", thresh)

        key = cv2.waitKey(1)
        if key == 27:
            break

video.release()
cv2.destroyAllWindows()