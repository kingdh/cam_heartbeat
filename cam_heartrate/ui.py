from matplotlib.widgets import Button
import matplotlib.pyplot as plt
import cam_heartrate.imgProcess as imgProcess
import eventlet


class Player(object):
    def __init__(self):
        pass
    def start(self):
        pass
    def pause(self):
        pass
    def exit(self):
        pass

class ButtonHandler(object):
    def __init__(self, player):
        self.player = player
        # self.range_s, self.range_e, self.range_step = 0, 1, 0.005

    def pause(self, event):
        self.player.pause()

    def start(self, event):
        self.player.start()

    def exit(self, event):
        self.player.exit()


class MainWindow(Player):
    def __init__(self, video):
        super().__init__()
        self.buttonHandler = ButtonHandler(self)
        self.videoFile = False
        self.pauseFlag = False
        self.exitFlag = False

        self.colorSig = []  # Will store the average RGB color values in each frame's ROI
        self.heartRates = []  # Will store the heart rate calculated every 1 second
        # TODO: maximum heart rate data number. if that number is reached, we should do something. But now we do nothing
        self.max_hr_num = 10000
        self.reader = imgProcess.VideoReader(video)
        self.fig = plt.figure(figsize=(7,8))
        self.fig.canvas.mpl_connect('close_event', self.close)

        # plt.title("heart rate detection")
        plt.ion()
        ax_start = plt.axes([0.2, 0.05, 0.1, 0.075])
        ax_pause = plt.axes([0.31, 0.05, 0.1, 0.075])
        ax_exit = plt.axes([0.52, 0.05, 0.1, 0.075])
        self.startbtn = Button(ax_start, 'start')
        self.startbtn.on_clicked(self.buttonHandler.start)
        self.pausebtn = Button(ax_pause, 'pause')
        self.pausebtn.on_clicked(self.buttonHandler.pause)
        self.exitbtn = Button(ax_exit, 'exit')
        self.exitbtn.on_clicked(self.buttonHandler.exit)
        self.ax1 = plt.axes((0.05, 0.25, 0.4, 0.7))
        self.ax1.set_xlabel("video")
        self.ax2 = plt.axes((0.55, 0.25, 0.4, 0.7))
        self.ax2.set_xlabel('Time (sec)')
        self.ax2.set_ylabel("bpm")
        self.ax2.title.set_text('Heart Rate')
        self.thread = None
        self.running = False


    def pause(self):
        self.pauseFlag = True

    def clean(self):
        self.colorSig = []
        self.heartRates = []

    def close(self):
        self.postExit()
        try:
            self.exitFlag = True
        except Exception as e:
            print(e)

    def exit(self):
        print("main window exit is called..")
        # plt.ion()
        self.exitFlag = True
        print("44444444", self.exitFlag, id(self.exitFlag))
        import time
        # eventlet.sleep(10)
        # self.thread.wait()
        self.postExit()
        # plt.close(self.fig)  # will cause the exit be called again

    def postExit(self):
        self.clean()
        self.reader.stop()

    def show(self):
        previousFaceBox = None
        i=0
        while True:
            i+=1
            if i % 15 == 0:
                print("buffer size = ", self.reader.buffer.qsize())

            eventlet.sleep(0)
            if self.pauseFlag:
                self.clean()
                print("pause is received, i=",i)
                break
            print("11111111111", self.exitFlag, id(self.exitFlag))
            if self.exitFlag:
                print("exit is received, break the loop")
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
            self.heartRates.append(imgProcess.calcBpm(roi, self.colorSig))  # calculate heart rate here
            self.ax1.imshow(highlighted)
            # plt.show()
            plt.pause(0.01)
        # plt.off()
        print("show exit...")

    def start(self):
        if self.running:
            return
        else:
            self.running = True
        if not self.reader.started:
            self.reader.start()

        # plt.ion()
        self.pauseFlag = False
        print("exit is set to False")
        self.exitFlag = False
        self.thread = imgProcess.pool.spawn(self.show)
        print("main window start exit...", self.thread)
        # self.thread.wait()
        # plt.ioff()  # no video is displayed
        # plt.show()

    def wait(self):
        while True:
            imgProcess.pool.waitall()
            print("wait loop...............", self.exitFlag, id(self.exitFlag))
            if not self.exitFlag:
                print("loop2", self.exitFlag)
                plt.pause(1)
                self.running = False
                self.pauseFlag = False
                # plt.show()
            else:
                print("both threads exit...")
                break

# TODO:
#  1. realtime heartbeat measurement.
#  2. get FPS according to the video.
#  3. add timestamp with the read frame, so we can calculate the frame rate.


# window = MainWindow("/Users/jinhui/workspaces/heartrate/231A_Project/video/qijie.mp4")
window = MainWindow("/home/jinhui/workspaces/heartrate/231A_Project/video/qijie.mp4")
window.start()
window.wait()