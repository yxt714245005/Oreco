# -*- encoding: utf-8 -*-
import time
import os
import cv2
import numpy as np
from multiprocessing import Process
from PIL import Image, ImageFont, ImageDraw
from cut import cut_image_from_array
from revert import binary_for_specimen


# FONT = ImageFont.truetype("simsun.ttc", 28, index=1) 
FONTS_EN_PATH = './fonts/en'
FONTS_ZH_PATH = './fonts/zh'
FONTS_EN = os.listdir(FONTS_EN_PATH)
FONTS_ZH = os.listdir(FONTS_ZH_PATH)


datetime = time.strftime('%Y%m%d%H%M%S', time.localtime())
data_root = 'data.{}'.format(datetime)
train_dir = os.path.join(data_root, 'train')
val_dir = os.path.join(data_root, 'val')
train_file = os.path.join(data_root, 'train.txt')
val_file = os.path.join(data_root, 'val.txt')
labels_file = os.path.join(data_root, 'labels.txt')


width = 70
high = 70


def init():
    if not os.path.exists(data_root):
        os.makedirs(data_root)

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    if not os.path.exists(val_dir):
        os.makedirs(val_dir)


def get_zh3500(zh_file='data/zh3500.txt'):
    """
    获取3500个常用汉字
    >>> len(get_zh3500())
    3500
    """
    zh_data = open(zh_file).read()
    zh_data = zh_data.replace('\r\n', '').split(',')
    return [unichr(int(s, 16)) for s in zh_data]


def get_zh7000(zh_file='data/zh7000.txt'):
    """
    获取约7000个汉字
    >>> len(get_zh7000())
    6763
    """
    zh_data = open(zh_file, 'r').readlines()
    return [s.strip('\r\n').decode('utf-8') for s in zh_data]


def get_zh_yao(zh_file='data/zh-yao.txt'):
    """
    获取约7000个汉字
    >>> len(get_zh_yao())
    3765
    """
    zh_data = open(zh_file, 'r').readlines()
    return [s.strip('\r\n').decode('utf-8') for s in zh_data]


def get_en():
    """
    获取英文字符加数字
    >>> len(get_en())
    66
    """
    return [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd',
        'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
        'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
        'y', 'z',
        '*', ':', '(', ')', '.', ',', '"', '‘', '’', '“', '”'
    ]


def get_image(text, x, y, font, rotate=0, mod="L"):
    """
    获取图片画板
    >>> font_file = os.path.join(FONTS_EN_PATH, 'times.ttf')
    >>> font = ImageFont.truetype(font_file, 26)
    >>> img = get_image('A', 0, 0, font)
    >>> img.width
    32
    """
    img = Image.new(mod, (width, high), 0)
    draw = ImageDraw.Draw(img)
    draw.text((x, y), text, font=font, fill=256)
    if rotate != 0 and str(rotate).isdigit():
        return img.rotate(int(rotate))
    return img


def gen_labels_data(words, file_name):
    """
    生成标签文件
    >>> file_name = '/tmp/label.txt'
    >>> gen_labels_data(['A',], file_name)
    >>> os.path.exists(file_name)
    True
    """
    for word in words:
        open(file_name, 'a').write('{} \n'.format(word.encode('utf-8')))


def get_rotate_and_multi_size_image(text, font_type='en'):
    '''
    获取不同大小，不同旋转度的问题图片列表
    >>> images = get_rotate_and_multi_size_image('A')
    >>> type(images)
    <type 'generator'>
    '''
    # 倾斜度范围
    rotates = range(-5, 5)
    # 字体大小范围
    font_sizes = range(26, 50, 2)

    if font_type == 'en':
        fonts = [os.path.join(FONTS_EN_PATH, item) for item in FONTS_EN]
    else:
        fonts = [os.path.join(FONTS_ZH_PATH, item) for item in FONTS_ZH]

    for ro in rotates:
        for font_file in fonts:
            for size in font_sizes:
                # 二值化和非二值化两种图片
                for t in (0, 1):
                    font = ImageFont.truetype(font_file, size)
                    if t == 1:
                        img = get_image(text, 2, -1, font, ro, "RGB")
                        image_array = np.array(img)
                        image_array = binary_for_specimen(image_array)
                    else:
                        img = get_image(text, 2, -1, font, ro, "L")
                        image_array = np.array(img)

                    ret_img = cut_image_from_array(image_array)
                    yield (text, ret_img)


def gen_data_process(words, start, test_prop, font_type):
    """
    通过多进程调用，将多个字符生产样本数据
    """

    for word_index, word in enumerate(words):
        index = start + word_index
        print word, index

        images = get_rotate_and_multi_size_image(word, font_type=font_type)
        for i, (text, img) in enumerate(images):
            word_dir = os.path.join(train_dir, str(index))
            if not os.path.exists(word_dir):
                os.makedirs(word_dir)

            is_test = True if (i % test_prop == 0) else False
            if not is_test:
                image_path = os.path.join(word_dir, '{}_{}.jpg'.format(index, i))
                open(train_file, 'a').write('{}/{}_{}.jpg {} \n'.format(index, index, i, index))
            else:
                image_path = os.path.join(val_dir, '{}_{}.jpg'.format(index, i))
                open(val_file, 'a').write('{}_{}.jpg {} \n'.format(index, i, index))
            cv2.imwrite(image_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def gen_image(test_prop=8, pro_num=20):

    all_process = list()
    en_data = get_en()
    zh_data = get_zh_yao()
    # zh_data = list(get_zh7000()[1001])

    gen_labels_data(en_data, labels_file)
    gen_labels_data(zh_data, labels_file)

    start = 0
    en_process = Process(target=gen_data_process, args=(en_data, start, test_prop, 'en'))
    all_process.append(en_process)

    zh_length = len(zh_data)
    pro_num = (pro_num - 1) if zh_length > (pro_num - 1)*10 else 1
    step = zh_length / pro_num

    start += len(en_data)
    for i in range(pro_num):
        if i != (pro_num-1):
            words = zh_data[step*i: step*(i+1)]
        else:
            words = zh_data[step*i:]

        zh_process = Process(target=gen_data_process, args=(words, start, test_prop, 'zh'))
        start += step

        all_process.append(zh_process)

    # 启动多进程处理
    for p in all_process:
        p.start()


if __name__ == '__main__':
    init()
    # word_list = ['你', '我', '他']
    # word_list = ['A', 'B', 'C']
    # word_list = get_zh('zh.txt')
    # word_list = get_en()
    gen_image(test_prop=8)
