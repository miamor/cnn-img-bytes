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
import api_img_bytes_config as cf
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
from tensorflow.python.keras.models import load_model, Model

from code_bytes.data_helpers import load_data_x

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
        return np.array(image)/255.0

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
        # print('np.array(image)', np.array(image).shape)
        return np.array(image)/255.0

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

        data = ' '.join(str(byte) for byte in hex_seq_data)[:cf.sequence_length*3]
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
                outdir_img = cf.__API_ROOT__ + '/api_tasks/__image{}/{}'.format(width, task_ids_str)
            else:
                outdir_img = cf.__API_ROOT__+'/api_tasks/__image/{}'.format(task_ids_str)
            outdir_seq = cf.__API_ROOT__+'/api_tasks/__seq/{}'.format(task_ids_str)

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

            q.put(rgb_data)
            q.put(seq_data)
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
            print('[Binaryy2Image][from_files] in queue', file_path)
            file_queue.put(file_path)

        # Start thread
        q = Queue()
        rgb_datas = []
        seq_datas = []
        # thread = Thread(target=self.run, args=(file_queue, task_ids, width, q))
        # thread.daemon = True
        # thread.start()
        for index in range(thread_number):
            thread = Thread(target=self.run, args=(file_queue, task_ids, width, q))
            thread.daemon = True
            thread.start()
        file_queue.join()
        while not q.empty():
            rgb_datas.append(q.get())
            seq_datas.append(q.get())
        
        return rgb_datas, seq_datas


class CNN_Img_Module:
    def __init__(self, img_model_path=None, cnn_bytes_model_path=None, lstm_bytes_model_path=None):
        self.bin2img = Binary2Image()

        print('[CNN_Img_Module] img_model_path', img_model_path)
        ''' Load model '''
        if img_model_path is not None:
            self.img_model = load_model(img_model_path)
        if cnn_bytes_model_path is not None:
            self.cnn_bytes_model = load_model(cnn_bytes_model_path)
        if lstm_bytes_model_path is not None:
            self.lstm_bytes_model = load_model(lstm_bytes_model_path)
        # model.summary()

        return
    
    def infer_img(self, X):
        print('[CNN_Img_Module][infer_img] X', X.shape)
        predict = self.img_model.predict(X)
        y_preds__img = predict.argmax(axis=1)
        print('predict', predict)
        return y_preds__img, predict
    
    def infer_bytes(self, X, model):
        print('[CNN_Img_Module][infer_bytes] X', X.shape)
        preds = model.predict(X)
        y_preds__bytes = np.array([1 if pred[0] > 0.5 else 0 for pred in preds])
        return y_preds__bytes, preds

    def from_files(self, filepaths, task_ids, output_directory=None):
        rgb_datas, seq_datas = self.bin2img.from_files(filepaths, task_ids=task_ids, width=64)
        print('[CNN_Img_Module][from_files] rgb_datas', len(rgb_datas))
        print('[CNN_Img_Module][from_files] \t  rgb_datas[0]', rgb_datas[0].shape)
        print('[CNN_Img_Module][from_files] seq_datas', len(seq_datas))
        print('[CNN_Img_Module][from_files] \t seq_datas[0]', len(seq_datas[0]))

        res = {}
        ''' Infer '''
        X_img = np.array(rgb_datas)
        res['img'] = self.infer_img(X_img)

        ''' CNN bytes '''
        X_bytes = load_data_x(seq_datas, sequence_length=cf.sequence_length, vocabulary_inv_path=cf.__API_ROOT__+'/code_bytes/data_prepared/len==200/vocabulary_inv.json')
        res['bytes_cnn'] = self.infer_bytes(X_bytes, self.cnn_bytes_model)

        ''' LSTM bytes '''
        res['bytes_lstm'] = self.infer_bytes(X_bytes, self.lstm_bytes_model)

        print('[CNN_Img_Module][from_files] res', res)
        return res

if __name__ == '__main__':

    filepaths = ['/home/mtaav/Desktop/old_uploads/85.0.4183.83_chrome_installer.exe',
                 '/home/mtaav/Desktop/old_uploads/clrcompression.dll']
    task_ids = [1,2]
    module = CNN_Img_Module(img_model_path=cf.__API_ROOT__+'/code_img/models/rgb.h5', cnn_bytes_model_path=cf.__API_ROOT__+'/code_bytes/output/cnn_best__7500_1259.h5', lstm_bytes_model_path=cf.__API_ROOT__+'/code_bytes/output/lstm_best__7240_1259.h5')
    module.from_files(filepaths, task_ids)
