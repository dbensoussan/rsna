import pandas as pd
from torch import save, load
from sklearn.model_selection import train_test_split

import os


def train_val_files(df, test_size, positive_rate=None, random_state=None):
    """
    Splits a dataframe in a train and a test (or validation) set with the possibility of
    removing negative observations to achieve a given percentage of positive observations.

    The resulting dataframes are saved in two separate csv files.
    """

    n_pos = df['any'].value_counts().loc[1]
    
    if positive_rate is not None:    
        n_neg = int(n_pos / positive_rate - n_pos)
        df_neg = df[df['any']==0].copy()
        
        try:
            df = df_neg.sample(n=n_neg, random_state=random_state).append(df[df['any']==1], ignore_index=True)
            df = df.sample(frac=1) # Shuffle rows
        except ValueError:
            raise ValueError('Positive rate in the dataframe must be higher than the initial rate of {:.3f}'.format(
            n_pos / len(df)))

    # Randomly splitting the data in two csv files
    df_train, df_val = train_test_split(df, test_size=test_size, random_state=random_state)
    df_train.to_csv('data/labels/train.csv', index=False)
    df_val.to_csv('data/labels/val.csv', index=False)
    
    print('Successfully created train (len: {}) and val (len: {}) sets'
          ' containing around {:.0f}% positives.'.format(len(df_train), len(df_val), n_pos * 100 / len(df)))
    
    
def save_model(model, optim, loss, model_name, dirname):
    path = os.path.join(dirname, '{}_{:.5f}.tar'.format(model_name, loss))
    save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'loss': loss,
    }, path)
    
    
def load_model(path, model, optim=None):
    checkpoint = load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optim:
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
    #loss = checkpoint['loss']
    
    
def create_prediction_csv(y, filename, dirname):
    path = os.path.join(dirname, filename)
    submission = pd.read_csv(os.path.join(dirname, 'stage_1_sample_submission.csv'))
    submission = pd.concat([submission.drop(columns=['Label']), pd.DataFrame(y)], axis=1)
    submission.columns = ['ID', 'Label']
    submission.to_csv('{}'.format(path), index=False)