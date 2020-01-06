import pandas as pd

# Preparing the test data
X = pd.read_csv('submissions/stage_1_sample_submission.csv')
X[['ID', 'Hem_Type']] = X['ID'].str.rsplit('_', n=1, expand=True)

X = X[['ID', 'Label']]
X.drop_duplicates(inplace=True) # one row per image
X.to_csv('data/labels/test.csv', index=False)