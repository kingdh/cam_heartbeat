import numpy as np
import cam_heartrate.imgProcess as imgProcess
from sklearn.decomposition import FastICA
import numpy.random
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.cluster import DBSCAN
import collections
import time


# NORMALIZE_SIG = False
NORMALIZE_SIG = True
MIN_HR_BPM = 45.0
MAX_HR_BMP = 240.0
SEC_PER_MIN = 60
MAX_FPS=MAX_HR_BMP/SEC_PER_MIN*2
CALC_INTERVAL = 1 # every such interval calculate bpm
# ICA = False
ICA = True
FREQ_PRECISION=2 #means 0.1**freq_precision
REVOLUTION = 10
FPS_ESTIMATION_PERIOD=2
BG_SAMPLE_RATIO = 0.4

def expand(a, b):
    d = (b - a) * 0.1
    return a-d, b+d

# class HeartBeatResult(object):
#     def __init__(self):
#         self.dist = 3
#         self.result = {}
#
#     def add(self, h):

class HeartDetector(object):
    def __init__(self):
        self.powerSpecs = []
        self.freqs = []
        self.WINDOW_SIZE = 256
        self.times = collections.deque([]) # init for fps estimation
        # self.t0 = time.time()
        self.counter = 0
        self.bins = {}
        self.last_ts = None
        self.bg = collections.deque([])
        self.bg_window = None
        # if imgProcess.HSV_MODE:
        #     self.bg = collections.deque([])
        # else:
        #     self.bg = None

    def read_hsv_signal(self,roi, timestamp, colorSigs, counter, fps, background, videoFile=False):
        if (roi is not None) and (np.size(roi) > 0):
            bg_idx = np.random.choice(background.shape[0], int(np.floor(background.shape[0]*BG_SAMPLE_RATIO)),replace=False)
            avg_bg_h = background[bg_idx].mean(axis=0)[2]
            colorChannels = roi.reshape(-1, roi.shape[-1])
            avgColor = colorChannels.mean(axis=0)[2]

            colorSigs.append(avgColor)
            self.bg.append(avg_bg_h)
            self.times.append(timestamp)

    def read_signal(self, roi, timestamp, window, colorSig, counter, fps, background, videoFile=False):
        if (roi is not None) and (np.size(roi) > 0):
            bg_idx = np.random.choice(background.shape[0], int(np.floor(background.shape[0]*BG_SAMPLE_RATIO)),replace=False)
            avg_bg = background[bg_idx].mean(axis=0)
            colorChannels = roi.reshape(-1, roi.shape[-1])
            avgColor = colorChannels.mean(axis=0)
            if imgProcess.GREEN_ONLY:
                avgColor = avgColor[1]
                avg_bg = avg_bg[1]
            window.append(avgColor)
            colorSig.append(avgColor)
            self.bg.append(avg_bg)
            self.bg_window.append(avg_bg)
            self.times.append(timestamp)
        calc = False
        if (self.times[-1] - self.last_ts) > CALC_INTERVAL:
            calc = True
            self.last_ts = self.times[-1]

        # print("len(colorSigs)=", len(colorSigs))
        if calc or (videoFile and counter == 0):
            if len(window) < self.WINDOW_SIZE:
                return 0, [0.,], [0.,], [0.,], [0.,]
            else:
                freqs, pwr = self._analyze_signal(window, fps, videoFile)
                bg_freqs, bg_pwr = self._analyze_signal(self.bg_window, fps, videoFile)
                assert(freqs.shape[0] == bg_freqs.shape[0])
                diff_pwr = pwr - bg_pwr
                heartRate = self.calc_bpm(freqs, diff_pwr)
                return heartRate, freqs, pwr, bg_freqs,bg_pwr
        else:
            return None,None, None, None, None

    def estimate_fps(self, roi, timestamp, reader):
        fps = 0
        if reader.videoFile and not imgProcess.DESAMPLE:
            fps = reader.fps
        elif (roi is not None) and (np.size(roi) > 0):
            # estimate camera fps, and return the window size
            colorChannels = roi.reshape(-1, roi.shape[-1])
            avgColor = colorChannels.mean(axis=0)
            self.times.append(timestamp)
            self.counter += 1
            if (self.times[-1] - self.times[0]) >= FPS_ESTIMATION_PERIOD:
                fps = self.counter / (self.times[-1] - self.times[0])
                print("estimated fps=", fps)
                if fps < MAX_FPS:
                    print("can't detect heart rate greater than ", fps*60/2, " /min")
        if fps > 0:
            self.counter = 0
            self.WINDOW_SIZE = fps / REVOLUTION * 60
            self.WINDOW_SIZE = int(np.exp2(np.ceil(np.log2(self.WINDOW_SIZE))))
            # self.WINDOW_SIZE = 512 # for test
            print("get window size=", self.WINDOW_SIZE)
            self.times = collections.deque([], self.WINDOW_SIZE)
            self.bg_window = collections.deque([], self.WINDOW_SIZE)
            self.last_ts = timestamp
            # if not imgProcess.HSV_MODE:
            #     self.bg = collections.deque([], self.WINDOW_SIZE)
            return self.WINDOW_SIZE

        return None

    def _analyze_signal(self, window, fps, videoFile=False):
        if not imgProcess.GREEN_ONLY and ICA:
            ica = FastICA(tol=0.3)
            sig = ica.fit_transform(window)
        else:
            sig = np.array(window)
        # interpolation to produce even distributed values
        even_times = np.linspace(self.times[0], self.times[-1], self.WINDOW_SIZE)
        if sig.ndim > 1:
            interpolated = np.array([np.interp(even_times, self.times, sig[:, i]) for i in range(sig.shape[-1])]).T
        else:
            interpolated = np.interp(even_times, self.times, sig)

        hamming_win = np.hamming(self.WINDOW_SIZE)
        if sig.ndim > 1:
            hamming_win = hamming_win.reshape((hamming_win.shape[0],1)).repeat(interpolated.shape[-1],axis=1)
        interpolated = hamming_win * interpolated
        sig = interpolated - np.mean(interpolated)

        if NORMALIZE_SIG:
            std = np.std(sig, axis=0)
            sig = sig/std
        if not videoFile:
            fps = float(self.WINDOW_SIZE) / (self.times[-1] - self.times[0])

        # Find power spectrum, use rfft instead of fft
        powerSpec = np.abs(np.fft.rfft(sig, axis=0)) ** 2

        freqs = np.fft.fftfreq(self.WINDOW_SIZE, 1.0 / fps)

        # Find heart rate
        validIdx = np.where((freqs >= MIN_HR_BPM / SEC_PER_MIN) & (freqs <= MAX_HR_BMP / SEC_PER_MIN))

        # TODO: maxPwrSrc is got from channels of R,G,B, is it right?
        if not imgProcess.GREEN_ONLY:
            maxPwrSrc = np.max(powerSpec, axis=1)
            validPwr = maxPwrSrc[validIdx]
        else:
            validPwr = powerSpec[validIdx]

        validFreqs = freqs[validIdx]
        # print("validFreqs=", validFreqs)

        # hr = self.calc_bpm(validFreqs, validPwr)

        return validFreqs, validPwr

    def append(self, freqs, power):
        """length of power and freq should be equal"""
        assert(len(power) == len(freqs))
        for tup in zip(freqs, power):
            bin = np.round(tup[0], decimals=FREQ_PRECISION) * 60
            if bin < MIN_HR_BPM:
                print("bin is out of range...")
            if bin in self.bins:
                self.bins[bin][1].append(tup[1]) #add pwr
                self.bins[bin][0].append(tup[0]) # add freq
            else:
                self.bins[bin] = [[tup[0], ], [tup[1],]]
        # self.powerSpecs.append(power)
        # self.freqs.append(freqs)
        self.counter += 1

    def calc_bpm(self,validFreqs, validPwr):
        self.append(validFreqs, validPwr)
        if self.counter < 3:
            maxPwrIdx = np.argmax(validPwr)
            hr = validFreqs[maxPwrIdx]
            return hr * 60

        tmp = {}
        for k, v in self.bins.items():
            tmp[k]=np.var(v[1])
        hr_var = {k: v for k, v in sorted(tmp.items(), key=lambda item: item[1], reverse=True)}
        # get top 5 freq
        i = 0
        result = []
        for k in hr_var:
            i += 1
            result.append(k)
            if i>= 7:
                break
        print("\n-------------------\n")
        print("possible hr=", result, ",\ncorresponding var:", [hr_var[f] for f in result])
        print("========using power to detect heartbeat=====")
        idx = np.argsort(validPwr)
        sorted_hr = [validFreqs[i]*60 for i in idx[len(idx):len(idx)-7:-1]]
        sorted_pwr = [validPwr[i] for i in idx[len(idx):len(idx)-7:-1]]
        print("possible hr=", sorted_hr, ",\ncorresponding pwr=", sorted_pwr)

        return result[0]


    def clear(self):
        self.times.clear()
        self.powerSpecs.clear()
        self.freqs.clear()

    def output(self):
        for bin in self.bins.items():
            print("\n----\nfreq=", bin[0], "; corresponding hr=", bin[0]*60)
            print("detail freqs=", bin[1][0])
            print("number of power spec", len(bin[1][1]), "; power spec=", bin[1][1])
            print("var(powerSpec)=", np.var(bin[1][1]))

        # print("powerspec=", self.powerSpecs)
        # print("powerSpec.var=", np.var(self.powerSpecs, axis=0))
        # print("freq=", self.freqs)


