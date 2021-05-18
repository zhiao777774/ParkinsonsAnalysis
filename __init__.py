import os
import time
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from keras.preprocessing.image import load_img, img_to_array


def initialize():
    cap = cv2.VideoCapture(0)
    time.sleep(1)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 630)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    print(cap.isOpened())
    if not cap.isOpened():
        cap.open(0)

    time.sleep(0.1)

    while True:
        frame = cap.read()[1]
        cv2.namedWindow('result', cv2.WINDOW_NORMAL)
        cv2.imshow('result', frame)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            # cv2.imwrite('/datasets/result.png', frame)
            
            cap.release()
            cv2.destroyAllWindows()
            break
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break

        

def plot():
    plt.style.use('dark_background')
    _DATA_PATH_STR = 'D:\\桌面\\Python\\parkinsons\\datasets\\drawings'
    _DATA_PATH_OBJ = Path(_DATA_PATH_STR)

    plt.figure(figsize=(12, 12))
    for i in range(1, 10, 1):
        plt.subplot(3, 3, i)
        img = load_img(f"{_DATA_PATH_STR}/spiral/training/healthy/" +
                       os.listdir(f"{_DATA_PATH_STR}/spiral/training/healthy")[i])
        plt.imshow(img)


    plt.figure(figsize=(12, 12))
    for i in range(1, 10):
        plt.subplot(3, 3, i)
        img = load_img(f"{_DATA_PATH_STR}/spiral/training/parkinson/" +
                       os.listdir(f"{_DATA_PATH_STR}/spiral/training/parkinson")[i])
        plt.imshow(img)

    plt.show()