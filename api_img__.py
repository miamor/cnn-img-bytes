# Binary to Image Converter
# Read executable binary files and convert them RGB and greyscale png images
#
# Author: Necmettin Çarkacı
# E-mail: necmettin [ . ] carkaci [ @ ] gmail [ . ] com

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import math
import api_img_config as cf
import tensorflow as tf
import numpy as np
import threading
import sys
import random
import os
from datetime import datetime
import argparse
from PIL import Image
from queue import Queue
from threading import Thread
from datetime import datetime


class Binary2Image:
    def getBinaryData(self, filepath):
        """
        Extract byte values from binary executable file and store them into list
        :param filepath: executable file name
        :return: byte value list
        """

        binary_values = []

        with open(filepath, 'rb') as fileobject:

            # read file byte by byte
            data = fileobject.read(1)

            while data != b'':
                binary_values.append(ord(data))
                data = fileobject.read(1)

        return binary_values

    def getHexData(self, filepath):
        """
        Extract byte values from binary executable file and store them into list
        :param filepath: executable file name
        :return: byte value list
        """
        hex_values = []

        with open(filepath, 'rb') as fileobject:

            # read file byte by byte
            data = fileobject.read(1)

            while data != b'':
                hex_values.append(data.hex())
                data = fileobject.read(1)

        return hex_values

    def createGreyScaleImage(self, filepath, width=None, outdir=None):
        """
        Create greyscale image from binary data. Use given with if defined or create square size image from binary data.
        :param filepath: image filepath
        """
        greyscale_data = self.getBinaryData(filepath)
        size = self.get_size(len(greyscale_data), width)
        image = Image.new('RGB', size)
        image.putdata(greyscale_data)
        if width > 0:
            image = image.resize((width, width))
        if outdir is not None:
            self.save_file(filepath, image, size, 'L', width, outdir)
        return np.array(image)

    def createRGBImage(self, filepath, width=None, outdir=None):
        """
        Create RGB image from 24 bit binary data 8bit Red, 8 bit Green, 8bit Blue
        :param filepath: image filepath
        """
        print('[createRGBImage] filepath, outdir', filepath, outdir)

        index = 0
        rgb_data = []

        # Read binary file
        binary_data = self.getBinaryData(filepath)

        # Create R,G,B pixels
        while (index + 3) < len(binary_data):
            R = binary_data[index]
            G = binary_data[index+1]
            B = binary_data[index+2]
            index += 3
            rgb_data.append((R, G, B))

        size = self.get_size(len(rgb_data), width)
        image = Image.new('RGB', size)
        image.putdata(rgb_data)
        if width > 0:
            image = image.resize((width, width))
        if outdir is not None:
            self.save_file(filepath, image, size, 'RGB', width, outdir)
        return np.array(image)

    def save_file(self, filepath, image, size, image_type, width, outdir='image'):
        try:
            # setup output filepath
            dirname = os.path.dirname(filepath)
            name, _ = os.path.splitext(filepath)
            name = os.path.basename(name)
            # imagename   = dirname + os.sep + image_type + os.sep + name + '_'+image_type+ '.png'
            imagename = outdir+'/'+name + '_'+image_type + '.png'
            os.makedirs(os.path.dirname(imagename), exist_ok=True)

            print('[save_file] size', size, (width, width), imagename)

            image.save(imagename)
            print('[save_file] The file', imagename, 'saved.')
        except Exception as err:
            print(err)

    def get_size(self, data_length, width=None):
        # source Malware images: visualization and automatic classification by L. Nataraj
        # url : http://dl.acm.org/citation.cfm?id=2016908

        if width is None:  # with don't specified any with value

            size = data_length

            if (size < 10240):
                width = 32
            elif (10240 <= size <= 10240 * 3):
                width = 64
            elif (10240 * 3 <= size <= 10240 * 6):
                width = 128
            elif (10240 * 6 <= size <= 10240 * 10):
                width = 256
            elif (10240 * 10 <= size <= 10240 * 20):
                width = 384
            elif (10240 * 20 <= size <= 10240 * 50):
                width = 512
            elif (10240 * 50 <= size <= 10240 * 100):
                width = 768
            else:
                width = 1024

            height = int(size / width) + 1

        else:
            width = int(math.sqrt(data_length)) + 1
            height = width

        return (width, height)

    def createSeq(self, filepath, outdir=None):
        hex_seq_data = self.getHexData(filepath)
        # print('[createSeq] hex_seq_data', hex_seq_data)

        data = ' '.join(str(byte) for byte in hex_seq_data)
        # print(data)
        if outdir is not None:
            # dirname = os.path.dirname(filepath)
            name, _ = os.path.splitext(os.path.basename(filepath))
            outname = outdir + '/' + name + '.txt'
            with open(outname, 'w') as f:
                f.write(data)
            print('[createSeq] File', outname, 'saved.')
        return data

    def run(self, file_queue, task_ids, width, q):
        task_ids = [str(id) for id in task_ids]
        task_ids_str = '-'.join(task_ids)
        # rgb_datas = []
        # seq_datas = []
        while not file_queue.empty():
            filepath = file_queue.get()
            if width is not None:
                outdir_img = cf.__IMG_API_ROOT__ + '/api_tasks/__image{}/{}'.format(width, task_ids_str)
            else:
                outdir_img = cf.__IMG_API_ROOT__+'/api_tasks/__image/{}'.format(task_ids_str)
            outdir_seq = cf.__IMG_API_ROOT__+'/api_tasks/__seq/{}'.format(task_ids_str)

            if not os.path.exists(outdir_seq):
                os.makedirs(outdir_seq, exist_ok=True)
            if not os.path.exists(outdir_img):
                os.makedirs(outdir_img, exist_ok=True)

            # createGreyScaleImage(filepath, width, outdir)
            rgb_data = self.createRGBImage(filepath, width, outdir_img)
            seq_data = self.createSeq(filepath, outdir=outdir_seq)
            # rgb_data = self.createRGBImage(filepath, width)
            # seq_data = self.createSeq(filepath)

            # rgb_datas.append(rgb_data)
            # seq_datas.append(seq_data)

            q.put((rgb_data, seq_data))
            file_queue.task_done()
        # return rgb_datas, seq_datas

    def from_folder(self, input_dir, width=None, thread_number=7):
        # Get all executable files in input directory and add them into queue
        file_queue = Queue()
        for root, directories, files in os.walk(input_dir):
            for filepath in files:
                print('in queue', filepath)
                file_path = os.path.join(root, filepath)
                file_queue.put(file_path)

        # Start thread
        for index in range(thread_number):
            thread = Thread(target=self.run, args=(file_queue, width))
            thread.daemon = True
            thread.start()
        file_queue.join()

    def from_files(self, filepaths, task_ids, width=None, thread_number=7):
        file_queue = Queue()
        for file_path in filepaths:
            print('in queue', file_path)
            file_queue.put(file_path)

        # Start thread
        q = Queue()
        datas = []
        # thread = Thread(target=self.run, args=(file_queue, task_ids, width, q))
        # thread.daemon = True
        # thread.start()
        for index in range(thread_number):
            thread = Thread(target=self.run, args=(file_queue, task_ids, width, q))
            thread.daemon = True
            thread.start()
        file_queue.join()
        while not q.empty():
            datas.append(q.get())
        
        return datas

class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(
            image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(
            self._decode_jpeg_data, channels=3)

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


tf.app.flags.DEFINE_integer('shards', 2, 'Number of shards in TFRecord files.')
tf.app.flags.DEFINE_integer('num_threads', 2, 'Number of threads to preprocess the images.')
FLAGS = tf.app.flags.FLAGS


class BuildImgData_RGB:
    def __init__(self):
        return

    def _int64_feature(self, value):
        """Wrapper for inserting int64 features into Example proto."""
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _bytes_feature(self, value):
        """Wrapper for inserting bytes features into Example proto."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _convert_to_example(self, filepath, image_buffer, height, width):
        """Build an Example proto for an example.
        Args:
        filepath: string, path to an image file, e.g., '/path/to/example.JPG'
        image_buffer: string, JPEG encoding of RGB image
        height: integer, image height in pixels
        width: integer, image width in pixels
        Returns:
        Example proto
        """

        colorspace = 'RGB'
        channels = 3
        image_format = 'JPEG'

        example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': self._int64_feature(height),
            'image/width': self._int64_feature(width),
            'image/colorspace': self._bytes_feature(tf.compat.as_bytes(colorspace)),
            'image/channels': self._int64_feature(channels),
            'image/format': self._bytes_feature(tf.compat.as_bytes(image_format)),
            'image/filepath': self._bytes_feature(tf.compat.as_bytes(os.path.basename(filepath))),
            'image/encoded': self._bytes_feature(tf.compat.as_bytes(image_buffer))}))
        return example

    def _is_png(self, filepath):
        # Determine if a file contains a PNG format image.
        return '.png' in filepath

    def _process_image(self, filepath, coder):
        # Process a single image file.

        # Read the image file.
        with tf.gfile.FastGFile(filepath, 'rb') as f:
            image_data = f.read()

        # Convert PNG to JPEG
        if self._is_png(filepath):
            print('Converting PNG to JPEG for %s' % filepath)
            image_data = coder.png_to_jpeg(image_data)

        # instance of ImageCoder to provide TensorFlow image coding utils.
        image = coder.decode_jpeg(image_data)

        # Check that image converted to RGB
        assert len(image.shape) == 3
        height = image.shape[0]
        width = image.shape[1]
        assert image.shape[2] == 3

        return image_data, height, width

    def _process_image_files_batch(self, coder, thread_index, ranges, name, filepaths, num_shards, output_directory):
        """Processes and saves list of images as TFRecord in 1 thread.
        Args:
        coder: instance of ImageCoder to provide TensorFlow image coding utils.
        thread_index: integer, unique batch to run index is within [0, len(ranges)).
        ranges: list of pairs of integers specifying ranges of each batches to
            analyze in parallel.
        name: string, unique identifier specifying the data set
        filepaths: list of strings; each string is a path to an image file
        num_shards: integer number of shards for this data set.
        """
        # Each thread produces N shards where N = int(num_shards / num_threads).
        # For instance, if num_shards = 128, and the num_threads = 2, then the first
        # thread would produce shards [0, 64).
        num_threads = len(ranges)
        assert not num_shards % num_threads
        num_shards_per_batch = int(num_shards / num_threads)

        shard_ranges = np.linspace(ranges[thread_index][0],
                                   ranges[thread_index][1],
                                   num_shards_per_batch + 1).astype(int)
        num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

        counter = 0
        for s in range(num_shards_per_batch):
            # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
            shard = thread_index * num_shards_per_batch + s
            output_filepath = '%s-%.5d-of-%.5d.tfrecord' % (
                name, shard, num_shards)
            output_file = os.path.join(output_directory, output_filepath)
            writer = tf.python_io.TFRecordWriter(output_file)

            shard_counter = 0
            files_in_shard = np.arange(
                shard_ranges[s], shard_ranges[s + 1], dtype=int)
            for i in files_in_shard:
                filepath = filepaths[i]

                image_buffer, height, width = self._process_image(
                    filepath, coder)

                example = self._convert_to_example(
                    filepath, image_buffer, height, width)
                writer.write(example.SerializeToString())
                shard_counter += 1
                counter += 1

                if not counter % 1000:
                    print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                          (datetime.now(), thread_index, counter, num_files_in_thread))
                    sys.stdout.flush()

            writer.close()
            print('%s [thread %d]: Wrote %d images to %s' %
                  (datetime.now(), thread_index, shard_counter, output_file))
            sys.stdout.flush()
            shard_counter = 0
        print('%s [thread %d]: Wrote %d images to %d shards.' %
              (datetime.now(), thread_index, counter, num_files_in_thread))
        sys.stdout.flush()

    def _process_image_files(self, name, filepaths, num_shards, output_directory):
        """Process and save list of images as TFRecord of Example protos.
        Args:
        name: string, unique identifier specifying the data set
        filepaths: list of strings; each string is a path to an image file
        num_shards: integer number of shards for this data set.
        output_directory: write processed image data
        """

        # Break all images into batches with a [ranges[i][0], ranges[i][1]].
        spacing = np.linspace(
            0, len(filepaths), FLAGS.num_threads + 1).astype(np.int)
        ranges = []
        for i in range(len(spacing) - 1):
            ranges.append([spacing[i], spacing[i+1]])

        # Launch a thread for each batch.
        print('Launching %d threads for spacings: %s' %
              (FLAGS.num_threads, ranges))
        sys.stdout.flush()

        # Create a mechanism for monitoring when all threads are finished
        coord = tf.train.Coordinator()

        # Create a generic TensorFlow-based utility for converting all image codings.
        coder = ImageCoder()

        threads = []
        for thread_index in range(len(ranges)):
            args = (coder, thread_index, ranges, name, filepaths, num_shards, output_directory)
            t = threading.Thread(target=self._process_image_files_batch, args=args)
            t.start()
            threads.append(t)

        # Wait for all the threads to terminate.
        coord.join(threads)
        print('%s: Finished writing all %d images in data set.' %
              (datetime.now(), len(filepaths)))
        sys.stdout.flush()

    def _find_image_files(self, data_dir, labels_file):
        """Build a list of all images files and labels in the data set.
        Returns:
        filepaths: list of strings; each string is a path to an image file.
        texts: list of strings; each string is the class, e.g. 'malware'
        """
        print('Determining list of input files from %s.' % data_dir)

        filepaths = []

        # Construct the list of JPEG files and labels.
        jpeg_file_path = '%s/*' % (data_dir)
        matching_files = tf.gfile.Glob(jpeg_file_path)

        filepaths.extend(matching_files)

        print('Found %d JPEG files inside %s.' % (len(filepaths), data_dir))
        return filepaths

    def _process_dataset(self, name, filepaths, num_shards, output_directory):
        # Process a complete data set and save it as a TFRecord.
        # Args:
        #   name: string, unique identifier specifying the data set.
        #   filepaths: list, list of images
        #   num_shards: integer number of shards for this data set.
        self._process_image_files(name, filepaths, num_shards, output_directory)

    def from_files(self, filepaths, task_ids, output_directory=None):
        assert not FLAGS.shards % FLAGS.num_threads, (
            'Please make the FLAGS.num_threads commensurate with '
            'FLAGS.shards')
        assert output_directory is not None, (
            'Please define output_directory')
        print('Saving results to %s' % output_directory)

        # Run it!
        task_ids = [str(id) for id in task_ids]
        name = '-'.join(task_ids)
        self._process_dataset(name, filepaths, FLAGS.shards, output_directory)

        # tf.app.run()


class CNN_Img_Module:
    def __init__(self):
        self.bin2img = Binary2Image()

        ''' Load model '''
        self.model = load_model('models/'+model_name+'.h5')
        # model.summary()

        return

    def prepare_data(self, filepaths, task_ids, output_directory=None):
        
        datas = self.bin2img.from_files(filepaths, task_ids=task_ids, width=64)
        print('datas', len(datas))
        rgb_datas = datas[0]
        seq_datas = datas[1]
        print('rgb_datas', len(rgb_datas))
        print('\t  rgb', rgb_datas[0].shape)
        print('seq_datas', len(seq_datas))


        ''' Infer '''
        predict = self.model.predict(rgb_datas)
        y_preds = predict.argmax(axis=1)


        # img_builder = BuildImgData_RGB()
        # img_builder.from_files(filepaths, task_ids, output_directory)


if __name__ == '__main__':

    # parser = argparse.ArgumentParser(
    #     prog='binar2image.py', description="Convert binary file to image")
    # parser.add_argument(
    #     dest='input_dir', help='Input directory path is which include executable files')

    # args = parser.parse_args()

    # main(args.input_dir, width=None)
    # bin2img.from_folder(args.input_dir, width=0)
    # main(args.input_dir, width=256)
    # main(args.input_dir, width=36)


    # task_ids = [1,2,3,4]

    # bin_filepaths = [cf.__IMG_API_ROOT__+'/api_tasks/bin/0a8ce026714e03e72c619307bd598add5f9b639cfd91437cb8d9c847bf9f6894',
    #                  cf.__IMG_API_ROOT__ +
    #                  '/api_tasks/bin/0c0a307a3f9882742406f44ddcccfd62448ce10cd7ceca1e377060d9b670b48b',
    #                  cf.__IMG_API_ROOT__ +
    #                  '/api_tasks/bin/2e70ea6467d4fef3c8ec276724fd95c6dd06e7ca5d8fdf4d79732bbcec904326',
    #                  cf.__IMG_API_ROOT__+'/api_tasks/bin/2ee99f7590d2eac42d729053f08f3bc2989e8ab3ebe49dc63d58a1dc4ed576d0']

    # bin2img = Binary2Image()
    # bin2img.from_files(bin_filepaths, task_ids=task_ids, width=0)


    # filepaths = []
    # dir = cf.__IMG_API_ROOT__+'/api_tasks/__image0/1-1/'
    # for file in os.listdir(dir):
    #     filepaths.append(dir+file)
    # task_ids = [1,1]

    # # filepaths = [cf.__IMG_API_ROOT__+'/api_tasks/RGB/0a8ce026714e03e72c619307bd598add5f9b639cfd91437cb8d9c847bf9f6894_RGB.png',
    # #              cf.__IMG_API_ROOT__ +
    # #              '/api_tasks/RGB/0c0a307a3f9882742406f44ddcccfd62448ce10cd7ceca1e377060d9b670b48b_RGB.png',
    # #              cf.__IMG_API_ROOT__ +
    # #              '/api_tasks/RGB/2e70ea6467d4fef3c8ec276724fd95c6dd06e7ca5d8fdf4d79732bbcec904326_RGB.png',
    # #              cf.__IMG_API_ROOT__+'/api_tasks/RGB/2ee99f7590d2eac42d729053f08f3bc2989e8ab3ebe49dc63d58a1dc4ed576d0_RGB.png']
    # img_builder = BuildImgData_RGB()
    # img_builder.from_files(filepaths, task_ids=task_ids, output_directory=cf.__IMG_API_ROOT__+'/api_tasks/__prepared_RGB')


    filepaths = ['/home/mtaav/Desktop/old_uploads/85.0.4183.83_chrome_installer.exe',
                 '/home/mtaav/Desktop/old_uploads/clrcompression.dll']
    task_ids = [1,2]
    module = CNN_Img_Module()
    module.prepare_data(filepaths, task_ids)
