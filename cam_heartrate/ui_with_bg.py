from matplotlib.widgets import Button
import matplotlib.pyplot as plt
from  matplotlib.animation import FuncAnimation
import cam_heartrate.imgProcess as imgProcess
import eventlet
import time
import numpy as np
from collections import deque
import cv2
from cam_heartrate.heartbeat_detector import HeartDetector
import cam_heartrate.heartbeat_detector as hd

class Player(FuncAnimation):
    def __init__(self, fig, func,  frames=None, init_func=None, fargs=None,
                  **kwargs):
        super().__init__(fig, func, frames, init_func, fargs, **kwargs)
        self.running = True
        # self.img = img

    def pause(self, event):
        self.running = False
        self.event_source.stop()

    def start(self, event):
        self.running = True
        self.event_source.start()


MAX_X_LIM = 10

def statistic(data):
    while True:
        tmp = data.popleft()
        if tmp != 0:
            data.append(tmp)
            break
    return np.mean(data), np.std(data)

class MainWindow(object):
    def __init__(self, video=None):
        super().__init__()
        self.exitFlag = False
        self.pauseFlag = False
        self.heartRates = deque([], MAX_X_LIM)  # Will store the heart rate calculated every 1 second
        self.timeline = deque([])
        # TODO: maximum heart rate data number. if that number is reached, we should do something. Now we do nothing
        self.max_hr_num = 10000

        self.reader = imgProcess.VideoReader(video)

        self.colorSig = deque([])
        self.window = None
        # self.fig = plt.figure(figsize=(12,8))
        self.fig = plt.figure(figsize=(8, 7))
        self.fig.canvas.mpl_connect('close_event', self.closed)
        self.player = Player(self.fig, self.show, frames=self.getFrames, interval=70)  # Animation
        # plt.title("heart rate detection")
        # plt.ion() //not used for animation
        ax_start = plt.axes([0.2, 0.05, 0.1, 0.055])
        ax_pause = plt.axes([0.31, 0.05, 0.1, 0.055])
        ax_exit = plt.axes([0.52, 0.05, 0.1, 0.055])
        self.startbtn = Button(ax_start, 'start')
        self.startbtn.on_clicked(self.player.start)
        self.pausebtn = Button(ax_pause, 'pause')
        self.pausebtn.on_clicked(self.player.pause)
        self.exitbtn = Button(ax_exit, 'exit')
        self.exitbtn.on_clicked(self.exit)

        self.ax1 = plt.axes((0.02, 0.57, 0.52, 0.38))
        self.ax1.set_axis_off()
        self.ax1.set_xlabel("video")
        self.ax2 = plt.axes((0.6, 0.57, 0.37, 0.38))
        self.ax2.set_xlabel('Time (sec)')
        self.ax2.set_ylabel("bpm")
        # self.ax2.set_ylim(140, 210)  #zhai, h*10 of hsv
        # self.ax2.set_ylim(2450, 2550) #zhai, v*15 of hsv
        self.ax2.set_xlim(0, MAX_X_LIM)
        # self.ax2.set_xlim(0, 20)
        self.ax2.title.set_text('Heart Rate')
        self.ax3 = plt.axes((0.1, 0.16, 0.7, 0.35))
        self.thread = None
        self.running = False
        self.img = None
        self.previousFaceBox = None
        self.start_time = None
        self.bpm = None
        self.bg = None
        self.fp = None
        self.bfp = None
        self.fp_x=[]
        self.fp_y=[]
        self.bfp_x=[]
        self.bfp_y=[]
        self.counter = 0
        self.x = np.arange(10)
        self.detector = HeartDetector()

    def clean(self):
        self.colorSig.clear()
        self.heartRates.clear()
        self.detector.clear()

    def closed(self, event):
        print("main window close is called...")
        self.postExit()
        try:
            self.exitFlag = True
            # print(self.colorSig)
            print("bpm=", self.heartRates)
            print("bpm mean=%f, bpm std=%f" % statistic(self.heartRates))
        except Exception as e:
            print(e)

    def exit(self, event):
        print("main window exit is called..")
        # self.detector.output()
        plt.close(self.fig)  # will cause self.closed() is called


    def postExit(self):
        # self.clean()
        self.reader.stop()

    def show(self, frame):

        if self.img is not None and frame is not None:
            self.img.set_array(frame)
            x = np.fromiter(self.timeline, float)
            y = np.fromiter(self.colorSig, float)
            left, right = self.ax2.get_xlim()
            if x[-1] >= right:
                right = np.ceil(x[-1]).astype(int)
                self.ax2.set_xlim(right-MAX_X_LIM, right)
            # print(x.shape,y.shape)
            # print("self.colorSig=", y[-1])
            self.bpm.set_data(x, y)
            self.bg.set_data(x, np.fromiter(self.detector.bg, float))
            # self.fp.set_data(self.fp_x,self.fp_y)
            # self.bfp.set_data(self.bfp_x, self.bfp_y)
        else:
            pass
        return self.img, self.bpm

    def getFrames(self):
        highlighted = None
        # t = time.time()
        while True:
            eventlet.sleep(0)
            if self.pauseFlag:
                self.clean()
                yield highlighted
            if self.exitFlag:
                print("exit is received, break the loop")
                return highlighted

            # Capture frame-by-frame
            try:
                frame, timestamp = self.reader.buffer.get(block=True, timeout=1)
            except Exception as e:
                # print("no frame is got from buffer ", e)
                yield None
            now = time.time()
            highlighted, roi, self.previousFaceBox, bg = imgProcess.highlightRoi(frame, self.previousFaceBox)
            if roi is None:
                print("don't detect any face......")
                yield highlighted
                continue
            if self.window is None:
                window_size = self.detector.estimate_fps(roi, timestamp, self.reader)
                if window_size is not None:
                    self.window = deque([], window_size)
                yield highlighted
                continue
            # Calculate heart rate every one second (once have 30-second of data)
            # bpm = imgProcess.calcBpm(roi, self.colorSig, self.counter, self.reader.fps, self.windowSize)
            # self.detector.read_hsv_signal(roi, timestamp, self.colorSig, self.counter, self.reader.fps, bg, self.reader.videoFile)
            bpm, self.fp_x,self.fp_y, self.bfp_x,self.bfp_y = self.detector.read_signal(roi, timestamp, self.window, self.colorSig, self.counter, self.reader.fps, bg, self.reader.videoFile)
            # print("len of signal=", len(self.colorSig))
            self.counter += 1
            if bpm is not None:
                self.heartRates.append(bpm)  # calculate heart rate here
            self.timeline.append(now - self.start_time)
            # reader's fps is only applied for video file, not for live show.
            if self.counter == np.ceil(self.reader.fps):
                self.counter = 0

            yield highlighted


    def start(self):
        if not self.reader.started:
            self.reader.start()
        frame, timestamp = self.reader.buffer.get(block=True, timeout=0.5)
        highlighted, roi, self.previousFaceBox, bg = imgProcess.highlightRoi(frame, self.previousFaceBox)
        # at first, we only show the video, but don't calculate the bpm
        self.detector.read_hsv_signal(roi, timestamp, self.colorSig, self.counter, self.reader.fps, bg, self.reader.videoFile) #will not calculate bpm at the beginning
        #
        # self.counter += 1
        self.timeline.append(timestamp)
        self.heartRates.append(0)
        self.img = self.ax1.imshow(highlighted)
        self.start_time = time.time()
        # self.bpm, = self.ax2.plot(self.timeline, np.fromiter(self.colorSig, float), label='head')
        self.bpm, = self.ax2.plot(self.timeline, self.colorSig, 'g', label='head')
        print("self.colorSig=", self.colorSig[-1])
        self.bg, = self.ax2.plot(self.timeline, np.fromiter(self.detector.bg, float), 'r', label='background')
        # self.fp, = self.ax3.plot([0,0],'x-g', linewidth=2)
        # self.bfp, = self.ax3.plot([0,0],'o-r', linewidth=2)
        self.running = True
        plt.show()

    def wait(self):
        """will not be called with FuncAnimation"""
        while True:
            imgProcess.pool.waitall()
            if not self.exitFlag:
                plt.pause(1)
                self.running = False
                self.pauseFlag = False
            else:
                print("both threads exit...")
                break

# TODO:
#  1. realtime heartbeat measurement.
#  2. get FPS according to the video. :done
#  3. use green channel
#  4. try to average the r,g,b values togather: r+g+b/3
#  5. cv2.equalizehist() - seems not very useful.
#  6. try hamming window:
#             even_times = np.linspace(self.times[0], self.times[-1], L)
#             interpolated = np.interp(even_times, self.times, processed)
#             interpolated = np.hamming(L) * interpolated
#  7. use rfft instead of fft.
#  8. wavelet?
#  9. how to identify the noise such as head movement?

np.set_printoptions(suppress=True)

# window = MainWindow("/Users/jinhui/workspaces/heartrate/231A_Project/video/qijie2.mp4")
window = MainWindow("/home/jinhui/workspaces/heartrate/231A_Project/video/qijie.mp4")
# window = MainWindow("/home/jinhui/workspaces/heartrate/231A_Project/video/qijie2.mp4")
# window = MainWindow("/home/jinhui/workspaces/heartrate/231A_Project/video/zhai.mp4")
# window = MainWindow("/home/jinhui/workspaces/heartrate/231A_Project/video/mkp.mp4")
# window = MainWindow("/home/jinhui/workspaces/heartrate/231A_Project/video/zhai1.avi")
# window = MainWindow("/home/jinhui/workspaces/heartrate/231A_Project/video/android-1.mp4")
# window = MainWindow("/Users/jinhui/workspaces/heartrate/231A_Project/video/android-1.mp4")
# window=MainWindow()
window.start()
# window.wait()
