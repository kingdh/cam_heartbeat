import numpy as np
import cam_heartrate.imgProcess as imgProcess
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.cluster import DBSCAN
import collections
import time

WINDOW_SIZE=128
NORMALIZE_SIG = False
ICA = True

def expand(a, b):
    d = (b - a) * 0.1
    return a-d, b+d

class HeartDetector(object):
    def __init__(self):
        self.powerSpecs = []
        self.freqs = []
        self.times = collections.deque([], WINDOW_SIZE)
        # self.t0 = time.time()
        self.counter = 0

    def read_signal(self,roi, timestamp, colorSigs, counter, fps):
        if (roi is not None) and (np.size(roi) > 0):
            colorChannels = roi.reshape(-1, roi.shape[-1])
            avgColor = colorChannels.mean(axis=0)
            if imgProcess.GREEN_ONLY:
                avgColor = avgColor[1]
            colorSigs.append(avgColor)
            self.times.append(timestamp)
        # print("len(colorSigs)=", len(colorSigs))
        if counter == 0:
            if len(colorSigs) < WINDOW_SIZE:
                return 0
            else:
                heartRate = self._analyze_signal(colorSigs, fps)
                return heartRate * 60
        else:
            return None

    def _analyze_signal(self, window, fps):
        if not imgProcess.GREEN_ONLY and ICA:
            ica = FastICA()
            sig = ica.fit_transform(window)
        else:
            sig = window
        # interpolation to produce even distributed values
        even_times = np.linspace(self.times[0], self.times[-1], WINDOW_SIZE)
        interpolated = np.interp(even_times, self.times, sig)
        interpolated = np.hamming(WINDOW_SIZE) * interpolated
        sig = interpolated - np.mean(interpolated)

        if NORMALIZE_SIG:
            std = np.std(sig, axis=0)
            sig = sig/std

        fps = float(WINDOW_SIZE) / (self.times[-1] - self.times[0])

        # Find power spectrum, use rfft instead of fft
        powerSpec = np.abs(np.fft.rfft(sig, axis=0)) ** 2

        freqs = np.fft.fftfreq(WINDOW_SIZE, 1.0 / fps)

        # Find heart rate
        validIdx = np.where((freqs >= imgProcess.MIN_HR_BPM / imgProcess.SEC_PER_MIN) & (freqs <= imgProcess.MAX_HR_BMP / imgProcess.SEC_PER_MIN))

        # TODO: maxPwrSrc is got from channels of R,G,B, is it right?
        if not imgProcess.GREEN_ONLY:
            maxPwrSrc = np.max(powerSpec, axis=1)
            validPwr = maxPwrSrc[validIdx]
        else:
            validPwr = powerSpec[validIdx]

        validFreqs = freqs[validIdx]
        self.append(validPwr, validFreqs)
        maxPwrIdx = np.argmax(validPwr)
        hr = validFreqs[maxPwrIdx]
        return hr

    def append(self, power, freq):
        self.powerSpecs.extend(power)
        self.freqs.extend(freq)
        self.counter += 1

    def clear(self):
        self.times.clear()
        self.powerSpecs.clear()
        self.freqs.clear()

    def output(self):
        print("powerspec=", self.powerSpecs)
        print("freq=", self.freqs)

    def clustering_DBSCAN(self):
        params = ((0.2, 5), (0.2, 10), (0.2, 15), (0.3, 5), (0.3, 10), (0.3, 15))
        data = np.stack((self.powerSpecs,self.freqs), axis=1)
        # 数据2
        # t = np.arange(0, 2*np.pi, 0.1)
        # data1 = np.vstack((np.cos(t), np.sin(t))).T
        # data2 = np.vstack((2*np.cos(t), 2*np.sin(t))).T
        # data3 = np.vstack((3*np.cos(t), 3*np.sin(t))).T
        # data = np.vstack((data1, data2, data3))
        # # # 数据2的参数：(epsilon, min_sample)
        # params = ((0.5, 3), (0.5, 5), (0.5, 10), (1., 3), (1., 10), (1., 20))

        matplotlib.rcParams['font.sans-serif'] = ['SimHei']
        matplotlib.rcParams['axes.unicode_minus'] = False

        plt.figure(figsize=(9, 7), facecolor='w')
        plt.suptitle('DBSCAN聚类', fontsize=15)

        for i in range(6):
            eps, min_samples = params[i]
            model = DBSCAN(eps=eps, min_samples=min_samples)
            model.fit(data)
            y_hat = model.labels_

            core_indices = np.zeros_like(y_hat, dtype=bool)
            core_indices[model.core_sample_indices_] = True

            y_unique = np.unique(y_hat)
            n_clusters = y_unique.size - (1 if -1 in y_hat else 0)
            print(y_unique, '聚类簇的个数为：', n_clusters)

            # clrs = []
            # for c in np.linspace(16711680, 255, y_unique.size):
            #     clrs.append('#%06x' % c)
            plt.subplot(2, 3, i + 1)
            clrs = plt.cm.Spectral(np.linspace(0, 0.8, y_unique.size))
            print(clrs)
            for k, clr in zip(y_unique, clrs):
                cur = (y_hat == k)
                if k == -1:
                    plt.scatter(data[cur, 0], data[cur, 1], s=10, c='k')
                    continue
                plt.scatter(data[cur, 0], data[cur, 1], s=15, c=clr, edgecolors='k')
                plt.scatter(data[cur & core_indices][:, 0], data[cur & core_indices][:, 1], s=30, c=clr, marker='o',
                            edgecolors='k')
            x1_min, x2_min = np.min(data, axis=0)
            x1_max, x2_max = np.max(data, axis=0)
            x1_min, x1_max = expand(x1_min, x1_max)
            x2_min, x2_max = expand(x2_min, x2_max)
            plt.xlim((x1_min, x1_max))
            plt.ylim((x2_min, x2_max))
            plt.plot()
            plt.grid(b=True, ls=':', color='#606060')
            plt.title(r'$\epsilon$ = %.1f  m = %d，聚类数目：%d' % (eps, min_samples, n_clusters), fontsize=12)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
