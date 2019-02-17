# -*- coding: utf-8 -*-
import picamera
import picamera.array
import cv2

with picamera.PiCamera() as camera:
    with picamera.array.PiRGBArray(camera) as stream:
        # カメラの解像度を320x240にセット
        camera.resolution = (320, 240)
        # カメラのフレームレートを15fpsにセット
        camera.framerate = 15

        # ホワイトバランスをfluorescent(蛍光灯)モードにセット
        #camera.awb_mode = 'auto'
        #camera.awb_mode = 'sunlight'
        #camera.awb_mode = 'cloudy'
        #camera.awb_mode = 'shade'
        #camera.awb_mode = 'tungsten'
        camera.awb_mode = 'fluorescent'
        #camera.awb_mode = 'incandescent'
        #camera.awb_mode = 'flash'
        #camera.awb_mode = 'horizon'

        while True:
            # stream.arrayにBGRの順で映像データを格納
            camera.capture(stream, 'bgr', use_video_port=True)

            # system.arrayをウインドウに表示
            cv2.imshow('frame', stream.array)

            # 'q'を入力でアプリケーション終了
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # streamをリセット
            stream.seek(0)
            stream.truncate()

        cv2.destroyAllWindows()

