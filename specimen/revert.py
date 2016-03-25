# -*- encoding: utf-8 -*-

import sys
import os
import cv2
import numpy as np
from PIL import Image, ImageFilter
import scipy.misc 
from cut import cut_image_from_array


# 不同大小图片适应扩大倍数
EXPAND = {
    0: 7,
    1: 6,
    2: 4,
    3: 3,
    4: 2,
    5: 1.6,
    6: 1.5,
    7: 1.3,
    8: 1.2
}


def get_expand(width):
    """
    根据不同宽度获取不同的扩大倍数
    """
    num = width / 10
    if num in EXPAND.keys():
        return EXPAND.get(num)

    return 1


def surround(img_array):
    """
    图片扩展一个像素，减少贴近边缘的预测
    """
    height = img_array.shape[0] 
    width = img_array.shape[1]
    template = np.zeros((height+2, width+2))
    template[1: -1, 1: -1] = img_array
    return template
    

def binary(img_array, source_mean=150):
    """
    二值化图片
    """
    # TODO 优化最精确数值
    sigma = 7 if source_mean < 150 else 5
    sigma = 23
    #print source_mean
    #blur_img = cv2.GaussianBlur(img_array, (sigma, sigma), 1.5)
    blur_img = cv2.medianBlur(img_array, 7)
    cvt_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY)

    _, threshold_img = cv2.threshold(cvt_img, 0, 255, cv2.THRESH_OTSU)
    return threshold_img


def binary_for_specimen(img_array):
    """
    二值化图片样本
    """
    sigma = 1
    blur_img = cv2.GaussianBlur(img_array, (sigma, sigma), 0.5)
    cvt_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY)

    _, threshold_img = cv2.threshold(cvt_img, 0, 255, cv2.THRESH_OTSU)
    return threshold_img


def dilate(img_array, s=2):
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (s, s))
    dilate_img = cv2.dilate(img_array, element)
    return dilate_img


def revert_image_bak(image_path):
    """
    将一张灰度图片进行锐化、二值化、黑白变换、边界切割
    """
    img = Image.open(image_path)
    filter_img = img.filter(ImageFilter.EDGE_ENHANCE)
    img_array = np.array(filter_img)

    source_height = img_array.shape[0]
    source_width = img_array.shape[1]
    source_mean = img_array.mean()

    expand = get_expand(source_width)
    img_array = scipy.misc.imresize(img_array,
                                    (int(source_height*expand), int(source_width*expand)),
                                    'cubic')

    # dilate_img = dilate(img_array)

    threshold_img = binary(img_array, source_mean=source_mean)
    revert_array = cv2.bitwise_not(threshold_img)

    surround_img = surround(revert_array)
    cut_image = cut_image_from_array(surround_img)
    return cut_image


def revert_image(image_path):
    """
    将一张灰度图片进行锐化、二值化、黑白变换、边界切割
    """
    # img_array = cv2.imread(image_path)
    # img_array = cv2.fastNlMeansDenoising(img, None, 3, 7, 21)
    img = Image.open(image_path)
    filter_img = img.filter(ImageFilter.EDGE_ENHANCE)
    img_array = np.array(filter_img)

    img_array = cv2.fastNlMeansDenoising(img_array, None, 10, 7, 21)

    source_height = img_array.shape[0]
    source_width = img_array.shape[1]
    source_mean = img_array.mean()

    expand = get_expand(source_width)

    img_array = cv2.resize(img_array,
                           (int(source_width*expand), int(source_height*expand)),
                           interpolation=cv2.INTER_CUBIC)

    # dilate_img = dilate(img_array)

    threshold_img = binary(img_array, source_mean=source_mean)
    revert_array = cv2.bitwise_not(threshold_img)

    surround_img = surround(revert_array)
    cut_image = cut_image_from_array(surround_img)
    return cut_image


def is_symbol(image_path):
    img = Image.open(image_path)
    filter_img = img.filter(ImageFilter.EDGE_ENHANCE)
    img_array = np.array(filter_img)

    source_height = img_array.shape[0]
    source_mean = img_array.mean()

    threshold_img = binary(img_array, source_mean=source_mean)
    hard_img = threshold_img[:source_height/2+1, :]
    return True if hard_img.mean() > 235 else False



if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage: python revert.py source_image_file'
        exit(0)

    image_path = sys.argv[1]
    sub_name = os.path.basename(image_path).split('.')[0]
    image_array = revert_image(image_path)
    cv2.imwrite('test2.jpg', image_array, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
