# Binary to Image Converter
# Read executable binary files and convert them RGB and greyscale png images
#
# Author: Necmettin Çarkacı
# E-mail: necmettin [ . ] carkaci [ @ ] gmail [ . ] com

import os, math
import argparse
from PIL import Image
from queue import Queue
from threading import Thread


def getBinaryData(filename):
	"""
	Extract byte values from binary executable file and store them into list
	:param filename: executable file name
	:return: byte value list
	"""

	binary_values = []

	with open(filename, 'rb') as fileobject:

		# read file byte by byte
		data = fileobject.read(1)

		while data != b'':
			binary_values.append(ord(data))
			data = fileobject.read(1)

	return binary_values


def getHexData(filename):
	"""
	Extract byte values from binary executable file and store them into list
	:param filename: executable file name
	:return: byte value list
	"""

	hex_values = []

	with open(filename, 'rb') as fileobject:

		# read file byte by byte
		data = fileobject.read(1)

		while data != b'':
			hex_values.append(data.hex())
			data = fileobject.read(1)

	return hex_values


def createGreyScaleImage(filename, width=None, outdir=''):
	"""
	Create greyscale image from binary data. Use given with if defined or create square size image from binary data.
	:param filename: image filename
	"""
	greyscale_data  = getBinaryData(filename)
	size            = get_size(len(greyscale_data), width)
	save_file(filename, greyscale_data, size, 'L', width, outdir)


def createRGBImage(filename, width=None, outdir=''):
	"""
	Create RGB image from 24 bit binary data 8bit Red, 8 bit Green, 8bit Blue
	:param filename: image filename
	"""
	index = 0
	rgb_data = []

	# Read binary file
	binary_data = getBinaryData(filename)

	# Create R,G,B pixels
	while (index + 3) < len(binary_data):
		R = binary_data[index]
		G = binary_data[index+1]
		B = binary_data[index+2]
		index += 3
		rgb_data.append((R, G, B))

	size = get_size(len(rgb_data), width)
	save_file(filename, rgb_data, size, 'RGB', width, outdir)


def save_file(filename, data, size, image_type, width, outdir='image'):

	try:
		image = Image.new(image_type, size)
		image.putdata(data)

		# setup output filename
		dirname     = os.path.dirname(filename)
		name, _     = os.path.splitext(filename)
		name        = os.path.basename(name)
		# imagename   = dirname + os.sep + image_type + os.sep + name + '_'+image_type+ '.png'
		imagename   = outdir+'/'+os.path.basename(dirname)+'/'+image_type+'/' + name + '_'+image_type+ '.png'
		os.makedirs(os.path.dirname(imagename), exist_ok=True)

		print('size', size, (width, width))
		if width > 0:
			image = image.resize((width, width))

		image.save(imagename)
		print('The file', imagename, 'saved.')
	except Exception as err:
		print(err)


def get_size(data_length, width=None):
	# source Malware images: visualization and automatic classification by L. Nataraj
	# url : http://dl.acm.org/citation.cfm?id=2016908

	if width is None: # with don't specified any with value

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
		width  = int(math.sqrt(data_length)) + 1
		height = width

	return (width, height)


def createSeq(filename, outdir=''):
	hex_seq_data  = getHexData(filename)
	# print('hex_seq_data', hex_seq_data)

	dirname     = os.path.dirname(filename)
	name, _     = os.path.splitext(filename)
	name        = os.path.basename(name)
	outname   	= outdir+'/'+os.path.basename(dirname)+'/' + name + '.txt'

	data = ' '.join(str(byte) for byte in hex_seq_data)
	# print(data)
	with open(outname, 'w') as f:
		f.write(data)
	print('File', outname, 'saved.')


def run(file_queue, width):
	if width is not None:
		outdir = '../data/image{}'.format(width)
	else:
		outdir = '../data/image'
	
	while not file_queue.empty():
		filename = file_queue.get()


		# setup output filename
		dirname     = os.path.dirname(filename)
		name, _     = os.path.splitext(filename)
		name        = os.path.basename(name)

		imagename   = outdir+'/'+os.path.basename(dirname)+'/L/' + name + '_L.png'
		#print('\t imagename l', imagename)
		if not os.path.exists(imagename):
			createGreyScaleImage(filename, width, outdir)

		imagename   = outdir+'/'+os.path.basename(dirname)+'/RGB/' + name + '_RGB.png'
		#print('\t imagename rgb', imagename)
		if not os.path.exists(imagename):
			createRGBImage(filename, width, outdir)

		imagename   = '../data/seq/'+os.path.basename(dirname)+'/' + name + '.txt'
		#print('\t imagename seq', imagename)
		if not os.path.exists(imagename):
			createSeq(filename, outdir='../data/seq')
		file_queue.task_done()


def main(input_dir, width=None, thread_number=7):
	if width is not None:
		outdir = '../data/image{}'.format(width)
	else:
		outdir = '../data/image'

	# Get all executable files in input directory and add them into queue
	file_queue = Queue()
	for root, directories, files in os.walk(input_dir):
		for filename in files:

			# setup output filename
			dirname     = os.path.dirname(filename)
			name, _     = os.path.splitext(filename)
			name        = os.path.basename(name)
			
			p1 = outdir+'/'+os.path.basename(root)+'/L/' + name + '_L.png'
			p2 = outdir+'/'+os.path.basename(root)+'/RGB/' + name + '_RGB.png'
			p3 = '../data/seq/'+os.path.basename(root)+'/' + name + '.txt'
			if not os.path.exists(p1) or not os.path.exists(p2) or not os.path.exists(p3):
				print(p1, p2, p3)
				print('in queue', filename)
				file_path = os.path.join(root, filename)
				file_queue.put(file_path)

	# Start thread
	for index in range(thread_number):
		thread = Thread(target=run, args=(file_queue, width))
		thread.daemon = True
		thread.start()
	file_queue.join()


if __name__ == '__main__':

	parser = argparse.ArgumentParser(prog='binary2image.py', description="Convert binary file to image")
	parser.add_argument(dest='input_dir', help='Input directory path is which include executable files', default='../../../MTAAV_data/bin/game_Linh')

	args = parser.parse_args()

	# main(args.input_dir, width=None)
	main(args.input_dir, width=0)
	# main(args.input_dir, width=256)
	# main(args.input_dir, width=36)

