import cam_heartrate.video_process.specularity as spc
import cv2
import numpy as np
import time


class ImageProcessor(object):
    def process(self, frames, **kwargs):
        pass

    def process1(self, frame, **kwargs):
        """only process one frame"""
        pass


class VideoLoader(object):
    def __init__(self, file):
        self.file = file
        if self.file is None:
            self.cap = cv2.VideoCapture(0)
        else:
            self.cap = cv2.VideoCapture(file)
        # frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width, self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("video frame size:", self.width, "*", self.height)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        print("fps=", self.fps)

    def read(self):
        # video_tensor = np.zeros((frame_count, height, width, 3), dtype='float')
        x = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret is True:
                print("read ", x, " frame")
                x += 1
                yield frame
            else:
                print("total ", x, " frames are read out")
                break

    def release(self):
        self.cap.release()

class VideoResizer(ImageProcessor):
    def __init__(self, ratio, ori_h=None, ori_w=None):
        self.ratio = ratio
        self.ori_w = ori_w
        self.ori_h = ori_h
        if self.ori_h is None:
            self.h = self.w = None
        else:
            self.h = int(np.round(self.ori_h * ratio))
            self.w = int(np.round(self.ori_w * ratio))

    def process(self, frames, **kwargs):
        for frame in frames:
            if self.w is None:
                shape = frame.shape
                self.ori_h, self.ori_w = shape[0], shape[1]
                self.h = int(np.round(self.ori_h * self.ratio))
                self.w = int(np.round(self.ori_w * self.ratio))

            resized = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            # cv2.resize(frame, (self.h, self.w), interpolation=cv2.INTER_AREA)
            yield resized


class FrameDesampler(ImageProcessor):
    def __init__(self, ori_fps, target_fps):
        self.ori_fps = ori_fps
        self.target_fps = target_fps
        assert(target_fps < ori_fps)
        self.step = int(np.round(self.ori_fps/self.target_fps))
        print("desample step = ", self.step)

    def process(self, frames, **kwargs):
        i = 0
        for frame in frames:
            i += 1
            if i < self.step:
                continue
            yield frame
            i = 0

class FrameRotator(ImageProcessor):
    def __init__(self, rotation):
        if rotation == 90:
            self.rotation = cv2.ROTATE_90_CLOCKWISE
        elif rotation == 180:
            self.rotation = cv2.ROTATE_180
        elif rotation == 270:
            self.rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
        else:
            self.rotation = None

    def process(self, frames, **kwargs):
        for frame in frames:
            if self.rotation is not None:
                frame = cv2.rotate(frame, self.rotation)
            yield frame

class SpecularReflectRemoval(ImageProcessor):
    def process(self, frames, **kwargs):
        # gray_img = spc.derive_graym(impath)
        for frame in frames:
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
            yield telea


class VideoSaver(ImageProcessor):
    def __init__(self, output, fps):
        self.file = output
        self.fps = fps
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

    def process(self, frames, **kwargs):
        i = 1
        four_cc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        frame0 = frames.__next__()
        if frame0 is not None:
            [self.h, self.w] = frame0.shape[0:2]
            writer = cv2.VideoWriter(self.file, four_cc, self.fps, (self.w, self.h), 1)
            writer.write(frame0)
            for frame in frames:
                writer.write(frame)
                i += 1
            # [height, width] = video_tensor[0].shape[0:2]
            # writer = cv2.VideoWriter(self._out_file_name, four_cc, 30, (width, height), 1)
            # for i in range(0, video_tensor.shape[0]):
            print("total ", i, " frames are written")
            writer.release()


def remove_specular(file):
    ts = time.time()
    output = file[0:-4] + str(ts) + ".mp4"
    reader = VideoLoader(file)
    desampler = FrameDesampler(reader.fps, 7)
    resizer = VideoResizer(0.5)
    writer = VideoSaver(output, 7)
    removal = SpecularReflectRemoval()

    frames = reader.read()
    samples = desampler.process(frames)
    resized = resizer.process(samples)
    removed = removal.process(resized)
    writer.process(removed)
    # writer.process(resized)


def test_save(file):
    ts = time.time()
    output = file[0:-4] + str(ts) + ".mp4"
    resizer = VideoResizer(0.5)
    reader = VideoLoader(file)
    writer = VideoSaver(output, 29)
    frames = reader.read()
    resized = resizer.process(frames)
    writer.process(resized)


if __name__ == "__main__":
    # file = "/home/jinhui/workspaces/heartrate/231A_Project/video/zhai.mp4"
    file = "/Users/jinhui/workspaces/heartrate/231A_Project/video/zhai.mp4"
    remove_specular(file)
    # test_save(file)