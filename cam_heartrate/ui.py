from matplotlib.widgets import Button
import matplotlib.pyplot as plt
import cam_heartrate.imgProcess as imgProcess
import eventlet


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
        self.buttonHandler = ButtonHandler(self)
        self.videoFile = False
        self.stopFlag = False
        self.exitFlag = False

        self.colorSig = []  # Will store the average RGB color values in each frame's ROI
        self.heartRates = []  # Will store the heart rate calculated every 1 second
        # TODO: maximum heart rate data number. if that number is reached, we should do something. But now we do nothing
        self.max_hr_num = 10000
        self.reader = imgProcess.VideoReader(video)
        self.fig = plt.figure()

        plt.title("heart rate detection")
        ax_start = plt.axes([0.2, 0.05, 0.1, 0.075])
        ax_stop = plt.axes([0.31, 0.05, 0.1, 0.075])
        ax_exit = plt.axes([0.52, 0.05, 0.1, 0.075])
        startbtn = Button(ax_start, 'start')
        startbtn.on_clicked(self.buttonHandler.start)
        stopbtn = Button(ax_stop, 'stop')
        stopbtn.on_clicked(self.buttonHandler.stop)
        exitbtn = Button(ax_exit, 'exit')
        exitbtn.on_clicked(self.buttonHandler.exit)
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
        # ax3 = self.fig.add_subplot(133)
        self.ax1.title.set_text('Video')
        self.ax2.title.set_text('Heart Rate')
        plt.xlabel('Time (sec)', fontsize=17, axes= self.ax2)
        plt.ylabel("bpm", fontsize=17, axes=self.ax2)
        plt.ion()

    def stop(self):
        self.stopFlag = True
        print("stop is received")

    def clean(self):
        self.colorSig = []
        self.heartRates = []

    def exit(self):
        self.exitFlag = True

    def postExit(self):
        self.clean()
        self.reader.stop()

    def show(self):
        previousFaceBox = None
        i=0
        while True:
            i+=1
            if i % 15 == 0:
                print("buffer size = ",self.reader.buffer.qsize())

            eventlet.sleep(0)
            if self.stopFlag:
                self.clean()
                break
            if self.exitFlag:
                self.postExit()
                break

            # Capture frame-by-frame
            try:
                frame = self.reader.buffer.get(block=True, timeout=0.5)
                # frame = self.reader.buffer.get_nowait()
            except Exception as e:
                print("no frame is got from buffer ", e)
                continue
            highlighted, roi, previousFaceBox = imgProcess.highlightRoi(frame, previousFaceBox)
            # Calculate heart rate every one second (once have 30-second of data)
            print("frame processed", len(self.colorSig))
            self.heartRates.append(imgProcess.calcBpm(roi, self.colorSig))  # calculate heart rate here

            self.ax1.imshow(highlighted)
            # plt.draw()

            plt.pause(0.01)

    def start(self):
        if not self.reader.started:
            self.reader.start()

        self.stopFlag = False
        self.exitFlag = False
        thread = imgProcess.pool.spawn(self.show())
        thread.wait()

    def wait(self):
        imgProcess.pool.waitall()

# TODO:
#  1. realtime heartbeat measurement.
#  2. get FPS according to the video.
#  3. add timestamp with the read frame, so we can calculate the frame rate.


window = MainWindow("/Users/jinhui/workspaces/heartrate/231A_Project/video/qijie.mp4")
window.start()
window.wait()