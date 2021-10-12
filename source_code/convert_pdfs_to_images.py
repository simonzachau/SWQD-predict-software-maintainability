import os
import tempfile
import PyPDF2
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
from subprocess import check_output


def convert_pdf_to_image(file_path):
	""" adapted from https://mtyurt.net/post/2019/multipage-pdf-to-jpeg-image-in-python.html """
	# save temp image files in temp dir, delete them after we are finished
	with tempfile.TemporaryDirectory() as temp_dir:
		# convert pdf to multiple image
		images = convert_from_path(file_path, output_folder=temp_dir)
		# save images to temporary directory
		temp_images = []
		for i in range(len(images)):
			image_path = str(i) + '.png'
			images[i].save(image_path, 'png')
			temp_images.append(image_path)
		# read images into pillow.Image
		imgs = list(map(Image.open, temp_images))
	# find minimum width of images
	min_img_width = min(i.width for i in imgs)
	# find total height of all images
	total_height = 0
	for i, img in enumerate(imgs):
		total_height += imgs[i].height
	# create new image object with width and total height
	merged_image = Image.new(imgs[0].mode, (min_img_width, total_height))
	# paste images together one by one
	y = 0
	for img in imgs:
		merged_image.paste(img, (0, y))
		y += img.height
	# delete temp images
	for img in temp_images:
		os.remove(img)
	return merged_image


def convert_all_pdfs_to_images(source_directory, target_directory):
	for input_path in Path(source_directory).glob('**/*.pdf'):
		output_path = str(input_path).replace(source_directory, target_directory) + '.png'
		
		# create directory if it doesn't exist yet
		directory = output_path.rsplit('/', 1)[0]
		if not os.path.exists(directory):
			os.makedirs(directory)

		image = resize(make_square(convert_pdf_to_image(input_path)))

		image.save(output_path, 'png')
		print('Successfully saved', output_path)


def normalize_all_pdfs(source_directory, target_directory):
	""" adpated from https://gist.github.com/dalgu90/9f62df70ac3462960c745cf673d3910c """
	# add as many blank pages to all pdfs such that they are all equally long
	largest_num_pages = 0
	for input_path in Path(source_directory).glob('**/*.pdf'):
		if get_num_pages(input_path) > largest_num_pages:
			print('largest numer of pages found:', get_num_pages(input_path), input_path)
		largest_num_pages = max(largest_num_pages, get_num_pages(input_path))

	for input_path in Path(source_directory).glob('**/*.pdf'):
		output_path = str(input_path).replace(source_directory, target_directory)

		# create directory if it doesn't exist yet
		directory = output_path.rsplit('/', 1)[0]
		if not os.path.exists(directory):
			os.makedirs(directory)

		reader = PyPDF2.PdfFileReader(open(input_path, 'rb'))
		writer = PyPDF2.PdfFileWriter()

		num_input_pages = get_num_pages(input_path)
		for i in range(num_input_pages):
			writer.addPage(reader.getPage(i))

		num_pages_needed = largest_num_pages - num_input_pages
		_, _, w, h = reader.getPage(0)['/MediaBox']
		for i in range(num_pages_needed):
			writer.addBlankPage(w, h)

		with open(output_path, 'wb') as out:
			writer.write(out)
			print('Successfully normalized', output_path)


def get_num_pages(pdf_path):
	""" adapted from https://stackoverflow.com/a/47169350 """
	# get number of pages of a pdf
	output = check_output(['pdfinfo', pdf_path]).decode()
	pages_line = [line for line in output.splitlines() if 'Pages:' in line][0]
	num_pages = int(pages_line.split(":")[1])
	return num_pages


def make_square(image):
	""" adapted from https://stackoverflow.com/a/44231784 """
	x, y = image.size
	size = max(x, y)
	new_image = Image.new('RGBA', (size, size), (255, 255, 255, 255))
	new_image.paste(image, (int((size - x) / 2), int((size - y) / 2)))
	return new_image


def resize(image, size=680):
	new_image = image.resize((size, size), Image.ANTIALIAS)
	return new_image


if __name__ == '__main__':
	convert_all_pdfs_to_images('dataset_PDFCode_results', 'dataset_images')
	# normalize_all_pdfs('dataset_PDFCode_results', 'dataset_PDFCode_results_normalized')
	# convert_all_pdfs_to_images('dataset_PDFCode_results_normalized', 'dataset_images_normalized')
