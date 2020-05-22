import cam_heartrate.video_process.specularity as spc
import cv2
import numpy as np
import time
import types


class VideoProcessor(object):
    def process(self, frames, context):
        j = 0
        for frame in frames:
            if j==0:
                self.do_context(context)
                j = 1
            else:
                yield self.process1(frame, context)

    def process1(self, frame, context):
        """only process one frame"""
        return frame

    def do_context(self, context):
        pass

class Processors(object):
    def __init__(self):
        self.context = {}
        self._processors = []
        self.reader = None

    def add_reader(self, reader):
        self.reader = reader
        return self

    def add(self, processor):
        self._processors.append(processor)
        return self

    def run(self):
        if self.reader is None:
            raise Exception("reader is None")

        frames = self.reader.read(self.context)
        for p in self._processors:
            frames = p.process(frames, self.context)
            if isinstance(frames, types.GeneratorType):
                continue
            else: # frame should be None
                assert (self._processors[-1] == p)
                break


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

    def read(self, context):
        context['ori_fps'] = self.fps
        context['fps'] = self.fps
        context['ori_w'] = self.width
        context['ori_h'] = self.height

        # video_tensor = np.zeros((frame_count, height, width, 3), dtype='float')
        x = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret is True:
                print("read ", x, " frame")
                x += 1
                yield frame, x - 1
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
            assert(start < 1.0)
            s = int(total * start)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, s)
        else:
            assert(type(start) == int and start < total)
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
        while self.cap.isOpened():
            if x == self.length:
                break
            ret, frame = self.cap.read()
            if ret is True:
                print("read ", x, " frame")
                x += 1
                yield frame, x - 1
            else:
                print("total ", x, " frames are read out")
                break
        self.release()

    def release(self):
        self.cap.release()


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

    def process1(self, frame, context):
        f, i = frame
            # if self.w is None:
            #     shape = frame.shape
            #     self.ori_h, self.ori_w = shape[0], shape[1]
            #     self.h = int(np.round(self.ori_h * self.ratio))
            #     self.w = int(np.round(self.ori_w * self.ratio))
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

    def process(self, frames, context):
        i = 0
        for frame, j in frames:
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

    def process(self, frames, context):
        for frame, i in frames:
            if self.rotation is not None:
                frame = cv2.rotate(frame, self.rotation)
            yield frame, i


class SpecularReflectRemoval(VideoProcessor):
    def process(self, frames, context):
        # gray_img = spc.derive_graym(impath)
        for frame, i in frames:
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
            yield telea, i


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

    def process(self, frames, context):
        i = 1
        four_cc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        frame0, j = frames.__next__()
        fps = context['fps']
        assert (fps is not None)
        if frame0 is not None:
            [self.h, self.w] = frame0.shape[0:2]
            writer = cv2.VideoWriter(self.file, four_cc, fps, (self.w, self.h), 1)
            writer.write(frame0)
            for frame, j in frames:
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

    def process(self, frames, context):
        for frame, j in frames:
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


if __name__ == "__main__":
    file = "/home/jinhui/workspaces/heartrate/231A_Project/video/zhai.mp4"
    # file = "/Users/jinhui/workspaces/heartrate/231A_Project/video/zhai.mp4"
    # remove_specular1(file)
    # test_save(file)
    get_img(file)
