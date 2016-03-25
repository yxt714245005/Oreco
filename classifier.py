# -*- encoding: utf-8 -*-

from __future__ import division
import argparse
import os
import time
import sys
import PIL.Image
import numpy as np
import scipy.misc
from config import CAFFE_ROOT
from google.protobuf import text_format
sys.path.insert(0, CAFFE_ROOT)
# Suppress most caffe output
os.environ['GLOG_minloglevel'] = '2'
import caffe
from caffe.proto import caffe_pb2


class RecEngine(object):
    def __init__(self, caffe_model, deploy_file, label_file, mean_file=None, use_gpu=False):
        """
        @params caffe_model str: .caffemodel 文件路径
        @params mean_file str: .binaryproto 均值文件路径
        @params labels_file: .txt 标签文件
        @use_gpu: if True, run inference on the GPU
        """
        self.caffe_model = caffe_model
        self.deploy_file = deploy_file
        self.mean_file = mean_file
        self.use_gpu = use_gpu

        if self.use_gpu:
            caffe.set_mode_gpu()

        self.net = self.get_net(self.caffe_model, self.deploy_file, self.use_gpu)
        self.transformer = self.get_transformer(self.deploy_file, mean_file)
        self.labels = self.read_labels(label_file)

    def load_image_by_array(self, image_array, height, width, color=True):
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
        img = image_array
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
            if color:
                img = np.tile(img, (1, 1, 3))
        elif img.shape[2] == 4:
            img = img[:, :, :3]

        img = scipy.misc.imresize(img, (height, width), 'bilinear')
        return img

    def get_net(self, caffe_model, deploy_file, use_gpu=True):
        """
        返回 caffe.Net 对象

        @params caffe_model str:  .caffemodel 文件
        @params deploy_file str: .prototxt 文件
        @params use_gpu bool:  if True, use GPU
        """
        if use_gpu:
            caffe.set_mode_gpu()

        # load a new model
        return caffe.Net(deploy_file, caffe_model, caffe.TEST)

    def get_transformer(self, deploy_file, mean_file=None):
        """
        返回 caffe.io.Transformer 对象

        @params deploy_file str: .prototxt 文件
        @params mean_file str: .binaryproto 文件(可选)
        """
        network = caffe_pb2.NetParameter()
        with open(deploy_file) as infile:
            text_format.Merge(infile.read(), network)

        if network.input_shape:
            dims = network.input_shape[0].dim
        else:
            dims = network.input_dim[:4]

        t = caffe.io.Transformer(inputs=dict(data=dims))
        # transpose to (channels, height, width)
        t.set_transpose('data', (2, 0, 1))

        # color images
        if dims[1] == 3:
            # channel swap
            t.set_channel_swap('data', (2, 1, 0))

        if mean_file:
            # set mean pixel
            with open(mean_file, 'rb') as infile:
                blob = caffe_pb2.BlobProto()
                blob.MergeFromString(infile.read())
                if blob.HasField('shape'):
                    blob_dims = blob.shape
                    assert len(blob_dims) == 4, 'Shape should have 4 dimensions - shape is "%s"' % blob.shape
                elif blob.HasField('num') and blob.HasField('channels') and \
                        blob.HasField('height') and blob.HasField('width'):
                    blob_dims = (blob.num, blob.channels, blob.height, blob.width)
                else:
                    raise ValueError('blob does not provide shape or 4d dimensions')
                pixel = np.reshape(blob.data, blob_dims[1:]).mean(1).mean(1)
                t.set_mean('data', pixel)

        return t

    def load_image(self, path, height, width, mode='RGB'):
        """
        从指定路径加载文件
        Returns： np.ndarray (channels x width x height)

        @params path str: 文件路径
        @params width str: 调整的宽度
        @params height: 调整的高度
        @params mode str: 图片类型 (RGB for color or L for grayscale)
        """
        image = PIL.Image.open(path)
        image = image.convert(mode)
        image = np.array(image)
        # squash
        image = scipy.misc.imresize(image, (height, width), 'bilinear')
        return image

    def forward_pass(self, images, net, transformer, batch_size=20):
        """
        获取每张图片的匹配概率
        return np.ndarray (nImages x nClasses)
        @params images list np.ndarrays: 图片数据列表
        @params net caffe.Net: 分类器
        @transformer caffe.io.Transformer: 转换器
        @batch_size int: 同时处理图片数量
        """
        caffe_images = []
        for image in images:
            if image.ndim == 2:
                caffe_images.append(image[:, :, np.newaxis])
            else:
                caffe_images.append(image)

        caffe_images = np.array(caffe_images)

        dims = transformer.inputs['data'][1:]

        scores = None
        for chunk in [caffe_images[x:x+batch_size] for x in xrange(0, len(caffe_images), batch_size)]:
            new_shape = (len(chunk),) + tuple(dims)
            if net.blobs['data'].data.shape != new_shape:
                net.blobs['data'].reshape(*new_shape)
            for index, image in enumerate(chunk):
                image_data = transformer.preprocess('data', image)
                net.blobs['data'].data[index] = image_data
            output = net.forward()[net.outputs[-1]]
            if scores is None:
                scores = np.copy(output)
            else:
                scores = np.vstack((scores, output))

        return scores

    def read_labels(self, labels_file):
        """
        Returns a list of strings

        Arguments:
        labels_file -- path to a .txt file
        """
        if not labels_file:
            print 'WARNING: No labels file provided. Results will be difficult to interpret.'
            return None

        labels = []
        with open(labels_file) as infile:
            for line in infile:
                label = line.strip()
                if label:
                    labels.append(label)
        assert len(labels), 'No labels found'
        return labels

    def classify_v0(self, image_files):
        """
        通过caffemodel进行分类
        @params image_file list: images 图片文件路径
        """
        _, channels, height, width = self.transformer.inputs['data']
        if channels == 3:
            mode = 'RGB'
        elif channels == 1:
            mode = 'L'
        else:
            raise ValueError('Invalid number for channels: %s' % channels)
        print _, channels, height, width
        images = [self.load_image(image_file, height, width, mode) for image_file in image_files]

        # Classify the image
        classify_start_time = time.time()
        scores = self.forward_pass(images, self.net, self.transformer)
        print 'Classification took %s seconds.' % (time.time() - classify_start_time,)

        # take top 5 results
        indices = (-scores).argsort()[:, :5]
        classifications = []
        for image_index, index_list in enumerate(indices):
            result = []
            for i in index_list:
                # 'i' is a category in labels and also an index into scores
                if self.labels is None:
                    label = 'Class #%s' % i
                else:
                    label = self.labels[i]
                result.append((label, round(100.0*scores[image_index, i], 4)))
            classifications.append(result)

        for index, classification in enumerate(classifications):
            print '{:-^80}'.format(' Prediction for %s ' % image_files[index])
            for label, confidence in classification:
                print '{:9.4%} - "{}"'.format(confidence/100.0, label)

    def classify(self, image_arrays, batch_size=20):
        """
        通过caffemodel进行分类

        @params image_arrays ndarray list: images 图片numpy数据列表
        """

        _, channels, height, width = self.transformer.inputs['data']

        images = [self.load_image_by_array(img, height, width) for img in image_arrays]

        # Classify the image
        scores = self.forward_pass(images, self.net, self.transformer, batch_size=batch_size)

        # take top 5 results
        indices = (-scores).argsort()[:, :5]
        classifications = []
        for image_index, index_list in enumerate(indices):
            result = []
            for i in index_list:
                # 'i' is a category in labels and also an index into scores
                if self.labels is None:
                    label = 'Class #%s' % i
                else:
                    label = self.labels[i]
                result.append((label, round(100.0*scores[image_index, i], 4)))
            classifications.append(result)

        return classifications


if __name__ == '__main__':
    script_start_time = time.time()

    parser = argparse.ArgumentParser(description='Classification example - DIGITS')

    ### Positional arguments

    parser.add_argument('caffemodel',   help='Path to a .caffemodel')
    parser.add_argument('deploy_file',  help='Path to the deploy file')
    parser.add_argument('image',        help='Path to an image')

    ### Optional arguments

    parser.add_argument('-m', '--mean',
                        help='Path to a mean file (*.npy)')
    parser.add_argument('-l', '--labels',
                        help='Path to a labels file')
    parser.add_argument('--gpu',
                        action='store_true',
                        help="use the GPU")

    args = vars(parser.parse_args())

    image_files = [args['image']]

    engine = RecEngine(args['caffemodel'],
                       args['deploy_file'],
                       args['labels'],
                       args['mean'],
                       args['gpu'])

    engine.classify_v0(image_files)

    print 'Script took %s seconds.' % (time.time() - script_start_time,)

