import pandas as pd

# Loading previously calculated image metadata
df_im_sum = pd.read_csv('data/filters/pixel_sum.csv')
df_black_pixels = pd.read_csv('data/filters/black_pixels.csv')

# Load and prepare labels
X = pd.read_csv('data/labels/stage_1_train.csv')
X = X.drop_duplicates(subset=['ID'])

X[['ID', 'Hem_Type']] = X['ID'].str.rsplit('_', n=1, expand=True)

X = X.merge(df_black_pixels, how='left', on='ID')
X = X.merge(df_im_sum, how='left', on='ID')

# As discovered in the eda notebook,
# we can throw away from the training set all the images that either have
# less than 100k pixel sum value or more than 250k black pixels
# as they do not contain enough information to detect an emorrhage.
X = X[(X['Black_pixels'] < 250000) & (X['Pixel_sum'] > 100000)]

X = X.pivot(index='ID', columns='Hem_Type', values='Label').reset_index()

X.to_csv('data/labels/train_filtered.csv', index=False)