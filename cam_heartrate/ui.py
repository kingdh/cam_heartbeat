from matplotlib.widgets import Button
import matplotlib.pyplot as plt
from  matplotlib.animation import FuncAnimation
import cam_heartrate.imgProcess as imgProcess
import eventlet
import time
import numpy as np
from collections import deque
import cv2

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


MAX_X_LIM = 60

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
        self.timeline = deque([], MAX_X_LIM)
        # TODO: maximum heart rate data number. if that number is reached, we should do something. Now we do nothing
        self.max_hr_num = 10000

        # self.reader = imgProcess.VideoReader(video, rotation=cv2.ROTATE_90_COUNTERCLOCKWISE)
        self.reader = imgProcess.VideoReader(video)

        self.windowSize = int(np.ceil(imgProcess.WINDOW_TIME_SEC * np.ceil(self.reader.fps)))
        self.colorSig = deque([], self.windowSize)
        # self.fig = plt.figure(figsize=(12,8))
        self.fig = plt.figure(figsize=(8, 7))
        self.fig.canvas.mpl_connect('close_event', self.closed)
        self.player = Player(self.fig, self.show, frames=self.getFrames, interval=70)  # Animation
        # plt.title("heart rate detection")
        # plt.ion() //not used for animation
        ax_start = plt.axes([0.2, 0.05, 0.1, 0.075])
        ax_pause = plt.axes([0.31, 0.05, 0.1, 0.075])
        ax_exit = plt.axes([0.52, 0.05, 0.1, 0.075])
        self.startbtn = Button(ax_start, 'start')
        self.startbtn.on_clicked(self.player.start)
        self.pausebtn = Button(ax_pause, 'pause')
        self.pausebtn.on_clicked(self.player.pause)
        self.exitbtn = Button(ax_exit, 'exit')
        self.exitbtn.on_clicked(self.exit)

        self.ax1 = plt.axes((0.05, 0.25, 0.3, 0.7))
        self.ax1.set_axis_off()
        self.ax1.set_xlabel("video")
        self.ax2 = plt.axes((0.4, 0.25, 0.5, 0.7))
        self.ax2.set_xlabel('Time (sec)')
        self.ax2.set_ylabel("bpm")
        self.ax2.set_ylim(0, 200)
        self.ax2.set_xlim(0, MAX_X_LIM)
        self.ax2.title.set_text('Heart Rate')
        self.thread = None
        self.running = False
        self.img = None
        self.previousFaceBox = None
        self.start_time = None
        self.bpm = 0
        self.counter = 0
        self.x = np.arange(10)

    def clean(self):
        self.colorSig.clear()
        self.heartRates.clear()

    def closed(self, event):
        print("main window close is called...")
        self.postExit()
        try:
            self.exitFlag = True
            print(self.colorSig)
            print("bpm=", self.heartRates)
            print("bpm mean=%f, bpm std=%f" % statistic(self.heartRates))
        except Exception as e:
            print(e)

    def exit(self, event):
        print("main window exit is called..")
        plt.close(self.fig)  # will cause self.closed() is called


    def postExit(self):
        # self.clean()
        self.reader.stop()

    def show(self, frame):
        if self.img is not None and frame is not None:
            self.img.set_array(frame)
            x = np.fromiter(self.timeline, float)
            y = np.fromiter(self.heartRates, float)
            left, right = self.ax2.get_xlim()
            if x[-1] >= right:
                right = np.ceil(x[-1]).astype(int)
                self.ax2.set_xlim(right-MAX_X_LIM, right)
            self.bpm.set_data(x, y)
        else:
            pass
        return self.img, self.bpm

    def getFrames(self):
        highlighted = None
        t = time.time()
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
                frame = self.reader.buffer.get(block=True, timeout=1)
            except Exception as e:
                # print("no frame is got from buffer ", e)
                yield None
            now = time.time()
            highlighted, roi, self.previousFaceBox = imgProcess.highlightRoi(frame, self.previousFaceBox)
            if roi is None:
                print("don't detect any face......")
                yield highlighted
            # Calculate heart rate every one second (once have 30-second of data)
            bpm = imgProcess.calcBpm(roi, self.colorSig, self.counter, self.reader.fps, self.windowSize)
            self.counter += 1
            if bpm is not None:
                self.heartRates.append(bpm)  # calculate heart rate here
                self.timeline.append(now - self.start_time)
            if self.counter == np.ceil(self.reader.fps):
                self.counter = 0
                # t1 = time.time()
                # print("24 frames used:", t1-t)
                # t = t1
            yield highlighted
            # matplotlib is not suitable for realtime plotting or video. consider PyQtGraph for an alternation.
            # self.ax1.imshow(highlighted)

            # cv2.imshow("video", highlighted)
            # plt.show()
            # plt.pause(0.01)
        # plt.off()
        # print("show exit...")

    # def start(self):
    #     if self.running:
    #         return
    #     else:
    #         self.running = True
    #     if not self.reader.started:
    #         self.reader.start()
    #
    #     self.pauseFlag = False
    #     print("exit is set to False")
    #     self.exitFlag = False
    #     self.thread = imgProcess.pool.spawn(self.show)
    #     print("main window start exit...", self.thread)
    #     # self.thread.wait()
        # plt.ioff()  # no video is displayed
        # plt.show()

    def start(self):
        if not self.reader.started:
            self.reader.start()
        frame = self.reader.buffer.get(block=True, timeout=0.5)
        highlighted, roi, self.previousFaceBox = imgProcess.highlightRoi(frame, self.previousFaceBox)
        self.heartRates.append(imgProcess.calcBpm(roi, self.colorSig, self.counter, self.reader.fps, self.windowSize))  # calculate heart rate here
        self.counter += 1
        self.timeline.append(0)
        self.img = self.ax1.imshow(highlighted)
        self.start_time = time.time()
        self.bpm, = self.ax2.plot(0,0)
        # self.bpm, = self.ax2.plot(self.x, 20*np.sin(np.random.randn(10)))
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

# window = MainWindow("/Users/jinhui/workspaces/heartrate/231A_Project/video/qijie.mp4")
window = MainWindow("/home/jinhui/workspaces/heartrate/231A_Project/video/qijie2.mp4")
# window = MainWindow("/home/jinhui/workspaces/heartrate/231A_Project/video/android-1.mp4")
# window = MainWindow("/Users/jinhui/workspaces/heartrate/231A_Project/video/android-1.mp4")
# window=MainWindow()
window.start()
# window.wait()
