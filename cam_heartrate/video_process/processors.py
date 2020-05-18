import cam_heartrate.video_process.specularity as spc
import cv2
import numpy as np
import time

class image_processor(object):
    def process(self, frames, **kwargs):
        pass
    def process1(self, frame, **kwargs):
        """only process one frame"""
        pass

class video_loader(object):
    def __init__(self, file):
        self.file = file
        self.cap = cv2.VideoCapture(file)
        # frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width, self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("video frame size:", self.width, "*", self.height)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

    def read(self):
        # video_tensor = np.zeros((frame_count, height, width, 3), dtype='float')
        x = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret is True:
                # video_tensor[x] = frame
                print("read ", x, " frame")
                x += 1
                yield frame
            else:
                print("total ", x, " frames are read out")
                break

class specular_reflect_removal(image_processor):
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

class video_saver(image_processor):
    def __init__(self, output, width, height):
        self.file = output
        four_cc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.writer = cv2.VideoWriter(self.file, four_cc, 30, (width, height), 1)

    def process(self, frames, **kwargs):
        for frame in frames:
            self.writer.write(frame)
        self.writer.release()


def remove_specular(file):
    ts = time.time()
    output = file[0:-4] + str(ts) + ".mp4"
    reader = video_loader(file)
    writer = video_saver(output, reader.width, reader.height)
    removal = specular_reflect_removal()

    frames = reader.read()
    removed = removal.process(frames)
    writer.process(removed)

if __name__ == "__main__":
    file = "/home/jinhui/workspaces/heartrate/231A_Project/video/zhai.mp4"
    remove_specular(file)