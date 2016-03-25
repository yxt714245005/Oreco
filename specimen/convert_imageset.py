# -*- encoding: utf-8 -*-
import os
import sys
import plyvel
import time
import numpy as np
from PIL import Image
from multiprocessing import Process
from config import CAFFE_ROOT
sys.path.insert(0, CAFFE_ROOT)
import caffe
import skimage


def load_image(filename, width, high, color=True):
    """
    Load an image converting from grayscale or alpha as needed.
    Parameters
    ----------
    filename : string
    color : boolean
        flag for color format. True (default) loads as RGB while False
        loads as intensity (if image is already grayscale).
    Returns
    -------
    image : an image with type np.float32 in range [0, 1]
        of size (H x W x 3) in RGB or
        of size (H x W x 1) in grayscale.
    """
    image = Image.open(filename).convert("L")
    image = image.resize((width, high))
    image_array = np.array(image)
    img = skimage.img_as_float(image_array).astype(np.float32)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        if color:
            img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def convert(image_set, source_dir,  db, resize_width, resize_height):

    wb = db.write_batch()
    count = 0
    for item in image_set:
        file, label = item.strip('\r\n').strip().split(" ")
        file_path = os.path.join(source_dir, file)
        image = load_image(file_path, resize_width, resize_height)

        # Reshape image
        image = image[:, :, (2, 1, 0)]
        image = image.transpose((2, 0, 1))
        image = image.astype(np.uint8, copy=False)

        # Load image into datum object
        datum = caffe.io.array_to_datum(image, int(label))

        wb.put('%08d_%s' % (count, file), datum.SerializeToString())

        count += 1
        if count % 1000 == 0:
            # Write batch of images to database
            wb.write()
            del wb
            wb = db.write_batch()
            print 'Processed %i images.' % count

    if count % 1000 != 0:
        # Write last batch of images
        wb.write()
        print '%s processed a total of %i images.' % (time.strftime('%H:%M:%S', time.localtime()),
                                                      count)
    else:
        print '%s processed a total of %i images.' % (time.strftime('%H:%M:%S', time.localtime()),
                                                      count)


def main(source_dir, label_file, level_db, resize_width=None, resize_height=None, pro_num=30):

    db = plyvel.DB(level_db, create_if_missing=True, error_if_exists=True, write_buffer_size=268435456)
    labels = open(label_file, 'r').readlines()

    all_process = list()
    pro_num = pro_num if len(labels) > pro_num*10 else 1
    step = len(labels) / pro_num
    for i in range(pro_num):
        if i != (pro_num-1):
            items = labels[step*i: step*(i+1)]
        else:
            items = labels[step*i:]

        process = Process(target=convert,
                          args=(items, source_dir, db, resize_width, resize_height))
        all_process.append(process)

    # 启动多进程处理
    for p in all_process:
        p.start()


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print 'Usage: python convert_imageset.py source_dir label_file db_file'
        exit(0)

    src_dir = sys.argv[1]
    label_file = sys.argv[2]
    db_file = sys.argv[3]
    main(src_dir,
         label_file,
         '{}_level_db'.format(db_file),
         resize_width=28,
         resize_height=28)