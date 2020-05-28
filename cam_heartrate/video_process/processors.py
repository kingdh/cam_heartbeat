import cam_heartrate.video_process.specularity as spc
import cv2
import numpy as np
import scipy.signal as signal
import numpy.linalg as linalg
from sklearn.decomposition import FastICA
import time
import types
import cam_heartrate.imgProcess as imgProcess


class VideoProcessor(object):
    def process(self, frames, context, offset=0):
        j = 0
        for frame in frames:
            if j == 0:
                self.do_context(context)
                j = 1
            else:
                yield self.process1(frame, context, offset)

    def process1(self, frame, context, offset=0):
        """only process one frame"""
        return frame if offset == 0 else frame[offset]

    def do_context(self, context):
        pass


class SignalProcessor(object):
    def process(self, signals, context, offset=0):
        raise NotImplementedError


class WindowProcessor(VideoProcessor):
    def __init__(self, w_size=64, overlap=1):
        self._window_size = w_size
        self._window = []
        self._time = []
        self._overlap = overlap
        self._counter = 0

    def enqueue(self, frame, ts):
        if len(self._window) < self._window_size:
            self._window.append(frame)
            self._time.append(ts)
            if len(self._window) == self._window_size:
                self._window = np.array(self._window)
                return True
        else:
            self._counter = (1 + self._counter) % self._overlap
            self._window[0:-1] = self._window[1:]
            self._window[-1] = frame
            self._time[0:-1] = self._time[1:]
            self._time[-1] = ts
            return self._counter == 0


class BatchWindowProcessor(WindowProcessor):
    def _process_window(self, context):
        """:returns: a tuple, which contains a window of processed signals and corresponding timestamps"""
        raise NotImplementedError

    def process_window(self, frames, context, offset):
        """output signals and times of windows_size once a time"""
        i = 0
        for x in frames:
            frame, j = x[offset] if offset != 0 else x
            i += 1
            if not isinstance(j, float):
                # already a window
                self._window = frame
                self._time = j
                yield self._process_window(context)
            if self.enqueue(frame, j):
                yield self._process_window(context)
        # last window
        if self._counter != 0:
            yield self._process_window(context)

    def process(self, frames, context, offset=0):
        windows = self.process_window(frames, context, offset)
        for window in windows:
            for frame in window:
                yield frame


class Processors(object):
    def __init__(self):
        self.context = {}
        self._processors = []
        self.reader = None
        self._frames = None
        self._forks = []
        self._offset = 0

    def add_reader(self, reader):
        self.reader = reader
        self._frames = self.reader.read(self.context)
        return self

    def input(self, frames, offset=0):
        self._frames = frames
        self._offset = offset

    def add(self, processor):
        try:
            n = processor.split
            forks = []
            for i in range(n-1):
                forks.append(Processors())
            self._forks.append(forks)
            self._processors.append(processor)
            return self, forks
        except AttributeError:
            self._processors.append(processor)
            return self

    def run(self):
        if self.reader is None:
            raise Exception("reader is None")
        forks = iter(self._forks)
        # frames = self.reader.read(self.context)
        frames = self._frames
        for p in self._processors:
            frames = p.process(frames, self.context, self._offset)
            self._offset = 0
            try:
                n = p.split
                f = next(forks)
                assert(n == len(f))
                for i in range(n-1):
                    f[i].input(frames, i+1)
            except AttributeError:
                pass

            if isinstance(frames, types.GeneratorType):
                continue
            else:  # frame should be None
                assert (self._processors[-1] == p)
                break

    def output(self):
        if self._frames is None:
            raise Exception("frame is None")
        frames = self._frames
        forks = iter(self._forks)
        # frames = self.reader.read(self.context)
        for p in self._processors:
            frames = p.process(frames, self.context, self._offset)
            self._offset = 0
            try:
                n = p.split
                f = next(forks)
                assert(n-1 == len(f))
                for i in range(n-1):
                    f[i].input(frames, i+1)
            except AttributeError:
                pass
        # return frames
        for f in frames:
            yield f


class VideoLoader(object):
    def __init__(self, filepath):
        self.file = filepath
        if self.file is None:
            self.cap = cv2.VideoCapture(0)
        else:
            self.cap = cv2.VideoCapture(filepath)
        # frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width, self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("video frame size:", self.width, "*", self.height)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        print("fps=", self.fps)
        if self.file is not None:
            self.slot = 1/self.fps

    def read(self, context):
        context['ori_fps'] = self.fps
        context['fps'] = self.fps
        context['ori_w'] = self.width
        context['ori_h'] = self.height

        # video_tensor = np.zeros((frame_count, height, width, 3), dtype='float')
        x = 0
        t0 = time.time()
        t = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret is True:
                print("read ", x, " frame")
                x += 1
                if self.file is None:
                    yield frame, time.time() - t0
                else:
                    t += self.slot
                    yield frame, t
            else:
                print("total ", x, " frames are read out")
                break

    def release(self):
        self.cap.release()


class VideoSegmentLoader(object):
    def __init__(self, filepath, start, length=None):
        """start: the begin point from which read the video. if it is a float, it means the proportion of the whole video, and
        if it is int, it is the frame index which begin from 0."""
        self.file = filepath
        if self.file is None:
            self.cap = cv2.VideoCapture(0)
        else:
            self.cap = cv2.VideoCapture(filepath)
        self.length = length
        total = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if type(start) == float:
            assert (start < 1.0)
            s = int(total * start)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, s)
        else:
            assert (type(start) == int and start < total)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        # frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width, self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("video frame size:", self.width, "*", self.height)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        print("fps=", self.fps)

    def read(self, context):
        context['ori_fps'] = self.fps
        context['fps'] = self.fps
        context['ori_w'] = self.width
        context['ori_h'] = self.height

        # video_tensor = np.zeros((frame_count, height, width, 3), dtype='float')
        x = 0
        t0 = time.time()
        while self.cap.isOpened():
            if x == self.length:
                break
            ret, frame = self.cap.read()
            if ret is True:
                print("read ", x, " frame")
                x += 1
                yield frame, time.time() - t0
            else:
                print("total ", x, " frames are read out")
                break
        self.release()

    def release(self):
        self.cap.release()


class RoiIdentifier(VideoProcessor):
    def __init__(self):
        self.previousFaceBox = None
        self.split = 2

    def process(self, frames, context, offset=0):  # todo: need to deal with the return values
        for x in frames:
            frame, j = x[offset] if offset != 0 else x
            self.previousFaceBox, roi, mask, bg = imgProcess.getBestROI(frame, self.previousFaceBox)
            yield (roi, j), (bg, j)


class FrameMean(VideoProcessor):
    """get mean value of the whole frame, not per channel"""

    def __init__(self, axis=None):
        """:arg axis default None means average value of all the channel."""
        self._axis = axis

    def process(self, frames, context, offset=0):
        for x in frames:
            frame, j = x[offset] if offset != 0 else x
            if (frame is not None) and (np.size(frame) > 0):
                color_channels = frame.reshape(-1, frame.shape[-1])
                avg_color = color_channels.mean(axis=self._axis) if self._axis is not None else color_channels.mean()
                yield avg_color, j


class Detrend(BatchWindowProcessor):
    def __init__(self, w_size, overlap=1, lamda=64):
        super(Detrend, self).__init__(w_size, overlap=overlap)
        dev2 = np.zeros((self._window_size - 2, self._window_size))
        np.fill_diagonal(dev2[:, :self._window_size - 2], 1)
        np.fill_diagonal(dev2[:, 1:self._window_size - 1], -2)
        np.fill_diagonal(dev2[:, 2:], 1)
        I = np.eye(self._window_size)
        self._cal = I - linalg.inv(I + lamda * lamda * np.dot(dev2.T, dev2))

    def _process_window(self, context):
        return np.dot(self._cal, self._window), self._time


class MovingAverage(BatchWindowProcessor):
    def __init__(self, w_size, overlap=1, ksize=3, t=3):
        super(MovingAverage, self).__init__(w_size, overlap=overlap)
        self._t = t
        self._ks = ksize

    def _process_window(self, context):
        for i in range(self._t):
            self._window = cv2.blur(self._window, ksize=self.ks)
        return self._window


class ICA(BatchWindowProcessor):
    def __init__(self, w_size, overlap=1, tol=0.3):
        super(ICA, self).__init__(w_size, overlap=overlap)
        self.tol = tol
        self.ica = FastICA(tol=0.3)

    def _process_window(self, context):
        return self.ica.fit_transform(self._window), self._time


class SigInterpolation(BatchWindowProcessor):
    def __init__(self, w_size, int_num, overlap=1):
        super(SigInterpolation, self).__init__(w_size, overlap=overlap)
        self._int_num = int_num

    def _process_window(self, context):
        """here frames should already be converted to signals
        :returns a series of windows of signals"""
        even_times = np.linspace(self._time[0], self.time[-1], self._int_num)
        context['fps'] = self._int_num / (self.time[-1] - self.time[0])
        signals = np.array(self._window)
        if signals.ndim > 1:
            interpolated = np.array(
                [np.interp(even_times, self._time, signals[:, i]) for i in range(signals.shape[-1])]).T
        else:
            interpolated = np.interp(even_times, self.times, signals)
        return interpolated, even_times


class HammingWindow(BatchWindowProcessor):
    def _process_window(self, context):
        hamming_win = np.hamming(self._window_size)
        assert (isinstance(self._window, np.ndarray))
        if self._window.ndim > 1:
            hamming_win = hamming_win.reshape((hamming_win.shape[0], 1)).repeat(self._window.shape[-1], axis=1)
        return hamming_win * self._window, self._time


class WindowNormalizer(BatchWindowProcessor):
    def _process_window(self, context):
        mean = np.mean(self._window, axis=0)
        std = np.std(self._window, axis=0)
        return (self._window - mean) / std, self._time


class ButterWorthBandPass(BatchWindowProcessor):
    def __init__(self, high, low, order=4, w_size=64, overlap=1):
        super(ButterWorthBandPass, self).__init__(w_size, overlap=overlap)
        self._h = high
        self._l = low
        self._order = order

    def _process_window(self, context):
        fps = context['fps']
        wn = [self._l * 2 / fps, self._h * 2 / fps]
        b, a = signal.butter(self._order, wn, 'bandpass')
        return signal.filtfilt(b, a, self._window), self._time


class FFT(BatchWindowProcessor):
    def _process_window(self, context):
        """
        :returns a tuple, which contains a sub-tuple of power spectrum and corresponding frequencies,
        and the timestamp"""
        fps = context['fps']
        context['window_size'] = self._window_size
        power_spec = np.abs(np.fft.rfft(self._window, axis=0)) ** 2
        freqs = np.fft.fftfreq(self._window_size, 1.0 / fps)
        return (power_spec, freqs), self._time


class VideoResizer(VideoProcessor):
    def __init__(self, ratio):
        self.ratio = ratio
        self.h = self.w = None

    def do_context(self, context):
        ori_w = context['ori_w']
        ori_h = context['ori_h']
        if ori_h is None:
            self.h = self.w = None
        else:
            self.h = int(np.round(ori_h * self.ratio))
            self.w = int(np.round(ori_w * self.ratio))

    def process1(self, frame, context, offset = 0):
        x = frame[offset] if offset != 0 else frame
        f, i = x
        resized = cv2.resize(f, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        # cv2.resize(frame, (self.h, self.w), interpolation=cv2.INTER_AREA)
        return resized, i


class FrameDesampler(VideoProcessor):
    def __init__(self, target_fps):
        self.target_fps = target_fps
        self.step = None

    def do_context(self, context):
        assert ('ori_fps' in context)
        ori_fps = context['ori_fps']
        assert (self.target_fps < ori_fps)
        self.step = int(np.round(ori_fps / self.target_fps))
        print("desample step = ", self.step)
        context['fps'] = self.target_fps

    def process(self, frames, context, offset=0):
        i = 0
        for x in frames:
            frame, j = x[offset] if offset != 0 else x
            if self.step is None:
                self.do_context(context)
            i += 1
            if i < self.step:
                continue
            yield frame, j
            i = 0


class FrameRotator(VideoProcessor):
    def __init__(self, rotation):
        if rotation == 90:
            self.rotation = cv2.ROTATE_90_CLOCKWISE
        elif rotation == 180:
            self.rotation = cv2.ROTATE_180
        elif rotation == 270:
            self.rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
        else:
            self.rotation = None

    def process(self, frames, context, offset=0):
        for x in frames:
            frame, j = x[offset] if offset != 0 else x
            if self.rotation is not None:
                frame = cv2.rotate(frame, self.rotation)
            yield frame, j


class SpecularReflectRemoval(VideoProcessor):
    def process(self, frames, context, offset=0):
        # gray_img = spc.derive_graym(impath)
        for x in frames:
            frame, j = x[offset] if offset != 0 else x
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            r_img = m_img = np.array(gray_img)

            rimg = spc.derive_m(frame, r_img)

            s_img = spc.derive_saturation(frame, rimg)

            spec_mask = spc.check_pixel_specularity(rimg, s_img)

            enlarged_spec = spc.enlarge_specularity(spec_mask)

            # use opencv's inpaint methods to remove specularity
            radius = 12
            telea = cv2.inpaint(frame, enlarged_spec, radius, cv2.INPAINT_TELEA)
            # ns = cv2.inpaint(frame, enlarged_spec, radius, cv2.INPAINT_NS)
            yield telea, j


MIN_HR_BPM = 45.0
MAX_HR_BMP = 240.0
SEC_PER_MIN = 60
FREQ_PRECISION = 2  # means 0.1**freq_precision


class MaxPowerSpec(SignalProcessor):
    def __init__(self, max_num=1):
        """pick the largest max_num power spectrum and corresponding frequencies"""
        self._num = max_num
        self._valid_idx = None

    def process(self, signals, context, offset=0):
        """":arg: signals should be the result of FFT.process() method"""
        for p, ts in signals:
            power_spec, freqs = p
            valid_idx = self._valid_idx if self._valid_idx is not None else np.where(
                (freqs >= MIN_HR_BPM / SEC_PER_MIN) & (freqs <= MAX_HR_BMP / SEC_PER_MIN))
            valid_pwr = power_spec[valid_idx]
            idx = np.argsort(valid_pwr)
            valid_freqs = freqs[valid_idx]
            sorted_freq = [valid_freqs[i] * 60 for i in idx[len(idx):len(idx) - self._num:-1]]
            sorted_pwr = [valid_pwr[i] for i in idx[len(idx):len(idx) - self._num:-1]]
            yield (sorted_freq, sorted_pwr), ts


class VideoSaver(VideoProcessor):
    def __init__(self, output):
        self.file = output
        self.w = None
        self.h = None
        self.writer = None

    # def process_bad(self, frames, **kwargs):
    #     i=0
    #     four_cc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    #     # four_cc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    #     # four_cc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    #     self.writer = cv2.VideoWriter(self.file, four_cc, self.fps, (self.w, self.h), 1)
    #     for frame in frames:
    #         self.writer.write(frame)
    #     self.writer.release()

    def process(self, frames, context, offset=0):
        i = 1
        four_cc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        frame0 = frames.__next__()
        frame0, j = frame0[offset] if offset != 0 else frame0
        fps = context['fps']
        assert (fps is not None)
        if frame0 is not None:
            [self.h, self.w] = frame0.shape[0:2]
            writer = cv2.VideoWriter(self.file, four_cc, fps, (self.w, self.h), 1)
            writer.write(frame0)
            for x in frames:
                frame, j = x[offset] if offset != 0 else x
                writer.write(frame)
                i += 1
            print("total ", i, " frames are written")
            writer.release()


class JpgSaver(VideoProcessor):
    def __init__(self, output):
        self.file = output
        self.w = None
        self.h = None
        self.writer = None

    def process(self, frames, context, offset=0):
        for x in frames:
            frame, j = x[offset] if offset != 0 else x
            cv2.imwrite(self.file, frame)


#
# def remove_specular(file):
#     ts = time.time()
#     output = file[0:-4] + str(ts) + ".mp4"
#     reader = VideoLoader(file)
#     desampler = FrameDesampler(reader.fps, 7)
#     resizer = VideoResizer(0.5)
#     writer = VideoSaver(output, 7)
#     removal = SpecularReflectRemoval()
#
#     frames = reader.read()
#     samples = desampler.process(frames)
#     resized = resizer.process(samples)
#     removed = removal.process(resized)
#     writer.process(removed)
#     # writer.process(resized)


def remove_specular1(file):
    ts = time.time()
    output = file[0:-4] + str(ts) + ".mp4"
    Processors().add_reader(VideoLoader(file)).add(FrameDesampler(7)).add(VideoResizer(0.5)).add(
        SpecularReflectRemoval()).add(VideoSaver(output)).run()


def get_img(file):
    ts = time.time()
    output = file[0:-4] + str(ts) + ".jpg"
    Processors().add_reader(VideoSegmentLoader(file, 100, 1)).add(
        SpecularReflectRemoval()).add(JpgSaver(output)).run()


def get_pwr(file):
    p1, p2 = Processors().add_reader(VideoLoader(file)).add(RoiIdentifier())
    sigs = p1.add(FrameMean(axis=0)).add(Detrend(2, lamda=2)).add(
        MovingAverage(2)).output()
    for sig in sigs:
        print(sig)


if __name__ == "__main__":
    file = "/home/jinhui/workspaces/heartrate/231A_Project/video/zhai.mp4"
    # file = "/Users/jinhui/workspaces/heartrate/231A_Project/video/zhai.mp4"
    # remove_specular1(file)
    # test_save(file)
    get_pwr(file)
