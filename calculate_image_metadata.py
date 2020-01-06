from os import walk
import pandas as pd
import matplotlib.image as mpimg

# Puts all file names in a list
f = []
for (dirpath, dirnames, filenames) in walk('data/images/stage_1_train_images_jpg/'):
    f.extend(filenames)
    break

# Counting pixel values of the images
im_sum = {im_id[:-4] : mpimg.imread('data/images/stage_1_train_images_jpg/{}.jpg'.format(im_id[:-4])).sum()
        for im_id in f}
pd.DataFrame.from_dict(im_sum, orient='index', columns=['Pixel_sum']).to_csv(
	'data/filters/pixel_sum.csv')

# Counting dark squares
dark_squares = {im_id[:-4] : (mpimg.imread('data/images/stage_1_train_images_jpg/{}.jpg'.format(im_id[:-4])) == 0).sum()
        for im_id in f}
pd.DataFrame.from_dict(dark_squares, orient='index', columns=['Black_pixels']).to_csv(
	'data/filters/black_pixels.csv')