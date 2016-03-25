# -*- encoding: utf-8 -*-

import numpy as np
import scipy.misc
from PIL import Image, ImageFont, ImageDraw


def get_critical(img_array):
    """
    获取图片白点边界值

    >>> img = Image.open('../test/0.jpg')
    >>> img_array = numpy.array(img)
    >>> get_critical(img_array)
    127
    """
    return img_array.max()/2


def get_around_point(img_array, critical):
    """
    获取被切割图片的四个点位置
    >>> img = Image.open('../test/0.jpg')
    >>> img_array = numpy.array(img)
    >>> critical = get_critical(img_array)
    >>> get_around_point(img_array, critical)
    3 22 4 23
    """
    shape_x = img_array.shape[1]
    shape_y = img_array.shape[0]
    x1, x2 = 0, shape_x
    y1, y2 = 0, shape_y

    x_points = np.where(img_array.max(0) > critical)[0]
    y_points = np.where(img_array.max(1) > critical)[0]
    if len(x_points) and len(y_points):
        x1 = x_points[0] - 1 if x_points[0] > 0 else x_points[0]
        x2 = x_points[-1] + 1 if x_points[-1] + 1 < shape_x else x_points[-1]

        y1 = y_points[0] - 1 if y_points[0] > 0 else y_points[0]
        y2 = y_points[-1] + 1 if y_points[-1] < shape_y else y_points[-1]

    return x1, x2+1, y1, y2+1


def cut_image(image, dist_file=''):
    """
    根据Image类型进行切割
    @return Image object
    >>> img = Image.open('../test/0.jpg')
    >>> new_img = cut_image(img)
    >>> new_img.width
    18
    >>> new_img.height
    19
    """
    img_array = np.array(image)
    critical = get_critical(img_array)
    x1, x2, y1, y2 = get_around_point(img_array, critical)
    box = (x1, y1, x2, y2)
    cut = image.crop(box)

    if dist_file:
        cut.save(dist_file)
    return cut


def cut_image_from_array(img_array, dist_file=''):
    """
    根据ndarray类型进行矩阵切割
    @return ndarray object
    >>> img = Image.open('../test/0.jpg').convert("L")
    >>> img_array = numpy.array(img)
    >>> new_img = cut_image_from_array(img_array)
    >>> new_img.shape
    (19, 19)
    """
    critical = get_critical(img_array)
    x1, x2, y1, y2 = get_around_point(img_array, critical)
    return img_array[y1: y2, x1: x2]


if __name__ == '__main__':
    # img = Image.open("test1.jpg").convert("L")
    # font = ImageFont.truetype("fonts/zh/simsun.ttc", 24)
    # img = Image.new('L', (30, 30), 0)
    # draw = ImageDraw.Draw(img)
    # draw.text((2, 0), 'A'.decode('utf-8'), font=font, fill=256)
    #
    # # print img_array
    # image_array = np.array(img)
    # new_img = cut_image_from_array(image_array)
    # scipy.misc.imsave('test2.jpg', new_img)
    # cut_image(img, 'test.jpg')
    import cv2
    image_array = cv2.imread("/tmp/0_10.jpg")
    img = cut_image_from_array(image_array)
    cv2.imwrite('test2.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
