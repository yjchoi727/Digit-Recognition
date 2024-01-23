import cv2
import numpy as np
from keras.models import load_model

model = load_model('C:/Users/727ke/Desktop/VRAR2/Marker_Based_AR/digit_recognition/models/tf-cnn-model.h5')

def get_numbers(y_pred):
    results = []
    for number, per in enumerate(y_pred[0]):
        if per != 0:
            final_number = str(int(number))
            per = round((per * 100), 2)
            results.append((final_number, per))
    return results  # 결과를 리스트로 반환

video = cv2.VideoCapture(0)

if video.isOpened():
    while True:
        check, img = video.read()
        img2 = img.copy()
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_gau = cv2.GaussianBlur(img_gray, (5, 5), 0)
        ret, thresh = cv2.threshold(img_gau, 120, 255, cv2.THRESH_BINARY_INV)  # threshold 값을 조절해보세요

        kernel = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(thresh, kernel, iterations=1)

        edged = cv2.Canny(dilation, 50, 250)

        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            for contour in contours:
                if cv2.contourArea(contour) > 1000:  # contour 넓이의 임계값을 조절해보세요
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

        cv2.imshow("Frame", img2)
        cv2.imshow("Contours Frame", thresh)

        key = cv2.waitKey(1)
        if key == 27:
            break

video.release()
cv2.destroyAllWindows()