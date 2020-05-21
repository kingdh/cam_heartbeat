import cv2
import numpy as np


def derive_graym(impath):
    ''' The intensity value m is calculated as (r+g+b)/3, yet
        grayscalse will do same operation!
        opencv uses default formula Y = 0.299 R + 0.587 G + 0.114 B
    '''
    # return cv2.imread(impath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    return cv2.imread(impath, cv2.IMREAD_GRAYSCALE)


def derive_m(img, rimg):
    ''' Derive m (intensity) based on paper formula '''

    # (rw, cl, ch) = img.shape
    # for r in range(rw):
    #     for c in range(cl):
    #         rimg[r, c] = int(np.sum(img[r, c]) / 3.0)
    #
    # return rimg
    rimg = np.mean(img, axis=2).astype('int')
    return rimg


def derive_saturation(img, rimg):
    ''' Derive staturation value for a pixel based on paper formula '''

    # s_img = np.array(rimg)
    s1 = img[:,:,0].astype(int) + img[:,:,2].astype(int)
    s2 = 2*img[:,:,1].astype(int)
    a = (1.5*(img[:,:,2] - rimg)).astype(int)
    b = (1.5*(rimg - img[:,:,0])).astype(int)
    s_img = np.where(s1-s2>=0, a, b)
    return s_img
    # (r, c) = s_img.shape
    # for ri in range(r):
    #     for ci in range(c):
    #         # opencv ==> b,g,r order
    #         s1 = int(img[ri, ci][0]) + int(img[ri, ci][2])
    #         s2 = 2 * img[ri, ci][1]
    #         if s1 >= s2:
    #             s_img[ri, ci] = 1.5 * (int(img[ri, ci][2]) - int(rimg[ri, ci]))
    #         else:
    #             s_img[ri, ci] = 1.5 * (int(rimg[ri, ci]) - int(img[ri, ci][0]))
    #
    # return s_img



def check_pixel_specularity(mimg, simg):
    ''' Check whether a pixel is part of specular region or not'''

    m_max = np.max(mimg) * 0.5
    s_max = np.max(simg) * 0.33
    spec_mask_0 = np.zeros(simg.shape, dtype=np.uint8)
    spec_mask_255 = np.empty(simg.shape, dtype=np.uint8)
    spec_mask_255.fill(255)
    spec_mask = np.where(np.logical_and(mimg>=m_max, simg<=s_max), spec_mask_255, spec_mask_0)
    # return spec_mask
    # (rw, cl) = simg.shape
    # spec_mask = np.zeros((rw, cl), dtype=np.uint8)
    # for r in range(rw):
    #     for c in range(cl):
    #         if mimg[r, c] >= m_max and simg[r, c] <= s_max:
    #             spec_mask[r, c] = 255

    return spec_mask


def enlarge_specularity(spec_mask):
    ''' Use sliding window technique to enlarge specularity
        simply move window over the image if specular pixel detected
        mark center pixel is specular
        win_size = 3x3, step_size = 1
    '''

    # win_size, step_size = (3, 3), 1
    # enlarged_spec = np.array(spec_mask)
    # for r in range(0, spec_mask.shape[0], step_size):
    #     for c in range(0, spec_mask.shape[1], step_size):
    #         # yield the current window
    #         win = spec_mask[r:r + win_size[1], c:c + win_size[0]]
    #
    #         if win.shape[0] == win_size[0] and win.shape[1] == win_size[1]:
    #             if win[1, 1] != 0:
    #                 enlarged_spec[r:r + win_size[1], c:c + win_size[0]] = 255 * np.ones((3, 3), dtype=np.uint8)

    # fast approach only for win_size=3
    tmp = np.ones(spec_mask.shape)
    tmp_mask = np.where(spec_mask==0, 0, tmp)  # only contains 0 or 1.
    b = np.diff(tmp_mask, axis=0)
    c = np.vstack((b, tmp_mask[[-1, ], :]))
    d = np.where(c == 1, c, tmp_mask)
    c = np.vstack((tmp_mask[[0, ], :], b))
    d = np.where(c == -1, 1, d)

    # vertical
    b = np.diff(tmp_mask, axis=1)
    c = np.hstack((b, tmp_mask[:, [-1, ]]))
    d = np.where(c == 1, c, d)
    c = np.hstack((tmp_mask[:, [0, ]], b))
    d = np.where(c == -1, 1, d)

    # diagonal, 135 degree
    # print(enlarged_spec)
    b = tmp_mask[:-1, :-1]
    # print(b)
    b = np.vstack((np.zeros((1, b.shape[1])), b))
    b = np.hstack((np.zeros((b.shape[0], 1)), b))
    # print(b)
    c = tmp_mask - b
    d = np.where(c == 1, c, d)
    d = np.where(c == -1, 1, d)
    # print(d)
    b = tmp_mask[1:, :-1]
    # print(b)
    b = np.vstack((b, np.zeros((1, b.shape[1]))))
    b = np.hstack((np.zeros((b.shape[0], 1)), b))
    c = tmp_mask - b
    d = np.where(c == 1, c, d)
    d = np.where(c == -1, 1, d)
    d = d*255
    enlarged_spec = np.where(d==255, d, spec_mask).astype('uint8')

    return enlarged_spec
