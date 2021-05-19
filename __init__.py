import sys
import signal
import time
import cv2
import numpy as np
from tensorflow.keras.models import load_model


def initialize():
    if len(sys.argv) < 2:
        print('Arguments: <mode>')
        sys.exit(1)
    mode = sys.argv[1]

    _signal_handle()
    if mode == 'camera':
        capture = cv2.VideoCapture(0)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 630)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        if not capture.isOpened():
            capture.open(0)

        time.sleep(.1)
        classifier = load_model('./model/')
        predict_result = None
        while True:
            frame = capture.read()[1]
            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('s'):
                cv2.imwrite(
                    f'./assets/result/{time.strftime("%Y%m%d-%H%M", time.localtime())}.png', frame)

                classes = {0: 'Healthy', 1: 'Parkinsons\'s'}
                pred = classifier.predict(frame)
                clas = classes[np.round(pred)]
                print(f'預測機率: {pred}')
                print(f'分類結果: {clas}')

                predict_result = {
                    'frame': frame,
                    'prob': pred,
                    'clas': clas
                }

                capture.release()
                cv2.destroyAllWindows()
                break

        cv2.text(predict_result.frame, f'分類結果: {predict_result.clas}',
                 (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (25, 25, 112), 1, cv2.LINE_AA)
        cv2.imshow('predict', predict_result.frame)
        while True:
            if not cv2.waitKey(1) is -1:
                cv2.destroyAllWindows()
                break
    elif mode == 'speech':
        print('speech')
    else:
        print('Arguments: <mode> must be "camera" or "speech".')


def _signal_handle():
    def _handler(signal, frame):
        cv2.destroyAllWindows()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
