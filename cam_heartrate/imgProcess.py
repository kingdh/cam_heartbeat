import cv2
import numpy as np
import random
import subprocess
import os
import eventlet
import eventlet.queue
import matplotlib.pyplot as plt
import re
from sklearn.decomposition import FastICA

# Toggle these for different ROIs
REMOVE_EYES = False
FOREHEAD_ONLY = False
USE_SEGMENTATION = False
USE_MY_GRABCUT = False
ADD_BOX_ERROR = False

MIN_FACE_SIZE = 100

WIDTH_FRACTION = 0.6 # Fraction of bounding box width to include in ROI
HEIGHT_FRACTION = 1

# TODO: FPS should be read from ffmpeg command output or somewhere.
# FPS = 14.99
FPS = 23.99
WINDOW_TIME_SEC = 30
WINDOW_SIZE = int(np.ceil(WINDOW_TIME_SEC * FPS))
MIN_HR_BPM = 45.0
MAX_HR_BMP = 240.0
MAX_HR_CHANGE = 12.0
SEC_PER_MIN = 60

SEGMENTATION_HEIGHT_FRACTION = 1.2
SEGMENTATION_WIDTH_FRACTION = 0.8
GRABCUT_ITERATIONS = 5
MY_GRABCUT_ITERATIONS = 2

EYE_LOWER_FRAC = 0.25
EYE_UPPER_FRAC = 0.5

BOX_ERROR_MAX = 0.5
MAX_FRAME_BUFF = 40000

pool = eventlet.GreenPool()
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def segment(image, faceBox):
    # if USE_MY_GRABCUT:
    #     foregrndMask, backgrndMask = grabCut(image, faceBox, MY_GRABCUT_ITERATIONS)
    #
    # else:
    mask = np.zeros(image.shape[:2], np.uint8)
    bgModel = np.zeros((1, 65), np.float64)
    fgModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, faceBox, bgModel, fgModel, GRABCUT_ITERATIONS, cv2.GC_INIT_WITH_RECT)
    backgrndMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), True, False).astype('uint8')

    backgrndMask = np.broadcast_to(backgrndMask[:, :, np.newaxis], np.shape(image))
    return backgrndMask

def getROI(image, faceBox):
    if USE_SEGMENTATION:
        widthFrac = SEGMENTATION_WIDTH_FRACTION
        heigtFrac = SEGMENTATION_HEIGHT_FRACTION
    else:
        widthFrac = WIDTH_FRACTION
        heigtFrac = HEIGHT_FRACTION

    # Adjust bounding box
    (x, y, w, h) = faceBox
    widthOffset = int((1 - widthFrac) * w / 2)
    heightOffset = int((1 - heigtFrac) * h / 2)
    faceBoxAdjusted = (x + widthOffset, y + heightOffset,
                       int(widthFrac * w), int(heigtFrac * h))

    # Segment
    if USE_SEGMENTATION:
        backgrndMask = segment(image, faceBoxAdjusted)

    else:
        (x, y, w, h) = faceBoxAdjusted
        backgrndMask = np.full(image.shape, True, dtype=bool)
        backgrndMask[y:y + h, x:x + w, :] = False

    (x, y, w, h) = faceBox
    if REMOVE_EYES:
        backgrndMask[y + h * EYE_LOWER_FRAC: y + h * EYE_UPPER_FRAC, :] = True
    if FOREHEAD_ONLY:
        backgrndMask[y + h * EYE_LOWER_FRAC:, :] = True

    roi = np.ma.array(image, mask=backgrndMask)  # Masked array
    return roi, backgrndMask

# Sum of square differences between x1, x2, y1, y2 points for each ROI
def distance(roi1, roi2):
    return sum((roi1[i] - roi2[i])**2 for i in range(len(roi1)))

def getBestROI(frame, faceCascade, previousFaceBox):
    # cv2.imshow("frame", frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1,
                                         minNeighbors=5, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
        # minNeighbors=5, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE), flags=cv2.CV_HAAR_SCALE_IMAGE)
    roi = None
    faceBox = None

    # If no face detected, use ROI from previous frame
    if len(faces) == 0:
        faceBox = previousFaceBox

    # if many faces detected, use one closest to that from previous frame
    elif len(faces) > 1:
        if previousFaceBox is not None:
            # Find closest
            minDist = float("inf")
            for face in faces:
                if distance(previousFaceBox, face) < minDist:
                    faceBox = face
        else:
            # Chooses largest box by area (most likely to be true face)
            maxArea = 0
            for face in faces:
                if (face[2] * face[3]) > maxArea:
                    faceBox = face

    # If only one face dectected, use it!
    else:
        faceBox = faces[0]

    if faceBox is not None:
        if ADD_BOX_ERROR:
            noise = []
            for i in range(4):
                noise.append(random.uniform(-BOX_ERROR_MAX, BOX_ERROR_MAX))
            (x, y, w, h) = faceBox
            x1 = x + int(noise[0] * w)
            y1 = y + int(noise[1] * h)
            x2 = x + w + int(noise[2] * w)
            y2 = y + h + int(noise[3] * h)
            faceBox = (x1, y1, x2-x1, y2-y1)

        # Show rectangle
        #(x, y, w, h) = faceBox
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)

        roi, mask = getROI(frame, faceBox)

    return faceBox, roi, mask

def changeFrame(frame, masked):
    """

    :param frame: origin frame
    :param roi:  is a masked array whose shape is same with frame.
    :return: a modified frame.
    """
    base = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

    selected = np.ma.array(base, mask=~masked)  # Masked array
    selected.mask[:,:,[0,2]] = True

    np.floor_divide(selected, 2, out=selected, where=~selected.mask)
    partialDimming = selected.data
    newBGR = cv2.cvtColor(partialDimming, cv2.COLOR_HLS2BGR)
    b, g, r = cv2.split(newBGR)
    changed = cv2.merge([r,g,b])
    return changed

def convert2MatplotColor(frame):
    b, g, r = cv2.split(frame)
    return cv2.merge([r,g,b])

def plotFrame(frame1, frame2):
    plt.figure(figsize=(9, 7), facecolor='w')
    ax1 = plt.subplot(121)
    b, g, r = cv2.split(frame1)
    plt.imshow(cv2.merge([r,g,b]))
    ax2 = plt.subplot(122)
    plt.imshow(frame2)
    ax1.title.set_text("original")
    ax2.title.set_text("highlight")
    plt.show()

def getRotationInfo(video):
    cmd = 'ffmpeg -i %s' % video
    rotation = -1
    try:
        p = subprocess.Popen(
            cmd.split(" "),
            stderr=subprocess.PIPE,
            close_fds=True
        )
        stdout, stderr = p.communicate()

        reo_rotation = re.compile('rotate\s+:\s(?P<rotation>.*)')
        decoded = stderr.decode("utf-8")
        match_rotation = reo_rotation.search(decoded)
        if match_rotation is None:
            return -1
        rotation = match_rotation.group("rotation")
    except FileNotFoundError:
        print("Waringing: can't get roation inforamtion")
        return -1

    if rotation == "90":
        return 90
    elif rotation == "180":
        return 180
    elif rotation == "270":
        return 270
    else:
        raise Exception("unknown rotation angle:" + rotation)

def getFrameRotation(videoFile):
    rotation = getRotationInfo(videoFile)

    if rotation == 90:
        rotation = cv2.ROTATE_90_CLOCKWISE
    elif rotation == 180:
        rotation = cv2.ROTATE_180
    elif rotation == 270:
        rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
    return rotation

def highlightRoi(frame, previousFaceBox):
    previousFaceBox, roi, mask = getBestROI(frame, faceCascade, previousFaceBox)
    changed = changeFrame(frame, mask)
    return changed, roi, previousFaceBox

def getHeartRate(window):
    # Normalize across the window to have zero-mean and unit variance
    mean = np.mean(window, axis=0)
    std = np.std(window, axis=0)
    normalized = (window - mean) / std

    # Separate into three source signals using ICA
    ica = FastICA()
    srcSig = ica.fit_transform(normalized)

    # Find power spectrum
    powerSpec = np.abs(np.fft.fft(srcSig, axis=0))**2
    freqs = np.fft.fftfreq(WINDOW_SIZE, 1.0 / FPS)

    # Find heart rate
    # TODO: maxPwrSrc is got from channels of R,G,B, is it right?
    maxPwrSrc = np.max(powerSpec, axis=1)
    validIdx = np.where((freqs >= MIN_HR_BPM / SEC_PER_MIN) & (freqs <= MAX_HR_BMP / SEC_PER_MIN))
    validPwr = maxPwrSrc[validIdx]
    validFreqs = freqs[validIdx]
    maxPwrIdx = np.argmax(validPwr)
    hr = validFreqs[maxPwrIdx]
    print(hr)

    # plotSignals(normalized, "Normalized color intensity")
    # plotSignals(srcSig, "Source signal strength")
    # plotSpectrum(freqs, powerSpec)

    return hr

def calcBpm(roi, colorSigs):
    if (roi is not None) and (np.size(roi) > 0):
        colorChannels = roi.reshape(-1, roi.shape[-1])
        avgColor = colorChannels.mean(axis=0)
        colorSigs.append(avgColor)

    if (len(colorSigs) >= WINDOW_SIZE) and (len(colorSigs) % np.ceil(FPS) == 0):
        windowStart = len(colorSigs) - WINDOW_SIZE
        window = colorSigs[windowStart: windowStart + WINDOW_SIZE]
        # lastHR = heartRates[-1] if len(heartRates) > 0 else None
        heartRate = getHeartRate(window)
        # print(heartRate)
        return heartRate * 60
    else:
        return 0

class VideoReader(object):
    def __init__(self, video, buffersize=MAX_FRAME_BUFF):
        if video is not None:
            # a file
            if not os.path.isfile(video):
                raise FileNotFoundError("%s is not found" % video)
            self.rotation = getFrameRotation(video)
            self.video = cv2.VideoCapture(video)
            self.videoFile = True
        else:
            self.video = cv2.VideoCapture(0)
        self.bufsize = buffersize
        self.buffer = eventlet.queue.LightQueue(buffersize)
        self.bufferUsed = False
        self.stopFlag = False
        self.thread = None
        self.started = False
        self.clear()

    def readFrames(self):
        """return a highlighted frame"""
        print("reader thread started...")
        i=1
        while True:
            eventlet.sleep(0)
            if self.stopFlag:
                break
            i+=1
            ret, frame = self.video.read()
            if not ret:
                print("didn't read a frame, reach the end of frame, size=", self.buffer.qsize())
                break
            else:
                try:
                    # block at most 1 second
                    if self.rotation >= 0:
                        frame = cv2.rotate(frame, self.rotation)
                    self.buffer.put(frame, True, 1)
                except eventlet.queue.Full:
                    print("queue is full!")
                    pass
        self.bufferUsed = True

    def clear(self):
        if self.bufferUsed:
            self.buffer = eventlet.queue.LightQueue(self.bufsize)
            self.stopFlag = False
            self.thread = None
            self.started = False
            self.bufferUsed = False

    def start(self):
        self.clear()
        self.thread = pool.spawn(self.readFrames)
        self.started = True

    def stop(self):
        self.stopFlag = True
        try:
            self.thread.cancel()
        except:
            pass
        self.video.release()
    # def clearBuffer(self):
    #     # just create a new buffer. in future we should clear the underlying deque with mutex.
    #     self.buffer = eventlet.queue.LifoQueue(self.bufsize)

def testOutPutFrame():
    framefile = "/home/jinhui/workspaces/heartrate/231A_Project/test1.png"
    frame = cv2.imread(framefile)
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    box, roi, mask = getBestROI(frame, faceCascade, None)
    changed = changeFrame(frame, mask)
    plotFrame(frame, changed)

# testOutPutFrame()





