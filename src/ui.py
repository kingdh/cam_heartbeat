from matplotlib import widgets
import matplotlib.pyplot as plt
import cv2
import os
import imgProcess
import numpy as np

STOP = 0
START = 1
EXIT = 2

class Player(object):
    def __init__(self):
        pass
    def start(self):
        pass
    def stop(self):
        pass
    def exit(self):
        pass

class ButtonHandler(object):
    def __init__(self, player):
        self.player = player
        # self.range_s, self.range_e, self.range_step = 0, 1, 0.005

    def stop(self, event):
        self.player.stop()

    def start(self, event):
        self.player.start()

    def exit(self, event):
        self.player.exit()


class MainWindow(Player):
    def __init__(self, video):
        super().__init__()
        self.buttonHandler = ButtonHandler()
        self.videoFile = False
        self.rotation = -1
        self.stopFlag = False
        self.exitFlag = False
        if video is not None:
            # a file
            if not os.path.isfile(video):
                raise FileNotFoundError("%s is not found" % video)
            self.rotation = imgProcess.getFrameRotation(video)
            self.video = cv2.VideoCapture(video)
            self.videoFile = True
        else:
            self.video = cv2.VideoCapture(0)

        self.faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        self.colorSig = []  # Will store the average RGB color values in each frame's ROI
        self.heartRates = []  # Will store the heart rate calculated every 1 second
        # TODO: maximum heart rate data number. if that number is reached, we should do something. But now we do nothing
        self.max_hr_num = 10000

    def init(self):
        pass

    def stop(self):
        self.stopFlag = True

    def clean(self):
        self.colorSig = []
        self.heartRates

    def exit(self):
        self.clean()
        self.exitFlag = True


    def postExit(self):
        self.clean()
        self.video.release()
        cv2.destroyAllWindows()

    def start(self):
        previousFaceBox = None
        self.stopFlag = False
        self.exitFlag = False

        while True:
            if self.stopFlag:
                self.clean()
                break
            if self.exitFlag:
                self.postExit()

            # Capture frame-by-frame
            ret, frame = self.video.read()
            if not ret:
                break

            highlighted, roi, previousFaceBox = imgProcess.highlightRoi(frame, self.faceCascade, self.rotation, previousFaceBox)

            imgProcess.calcHeartBeat(roi, self.colorSig, self.hear)
            # Calculate heart rate every one second (once have 30-second of data)
            print("frame processed", len(self.colorSig))
            self.heartRates.append(imgProcess.Bpm(roi, self.colorSig))  # calculate heart rate here
            cv2.imshow('ROI', roi)
            cv2.waitKey(1)


# TODO: 1. realtime heartbeat measurement. 2. get FPS according to the video.