import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds

tfds.disable_progress_bar()


def prepare_titanic(test_size=0.3, random_state=123):
    print('Download or read from disk.')
    ds = tfds.load('titanic', split='train')
    
    # Turn DataSet adapter into DataFrame
    print('Convert to pandas.DataFrame')
    X = []
    y = []
    for ex in tfds.as_numpy(ds):
        x_i, y_i = ex['features'], ex['survived']
        X.append(x_i)
        y.append(y_i)

    df_X = pd.DataFrame(X)
    features = list(df_X.columns)
    y = pd.Series(y, name='survived')
    
    print('Partition into Train and Test')
    print(f'    Test size = {test_size}')
    print(f'    random_state = {random_state}')
    df_train, df_test, y_train, y_test = train_test_split(
        df_X, y, test_size=test_size, random_state=random_state
    )
    
    return df_train, df_test, y_train, y_test

def encode_features(df):
    
    df = df.copy()
    
    # Columns requiring special handling
    df['cabin'] = np.where(df['cabin'] == b'Unknown', 1, 0)
    df['ticket'] = df['ticket'].astype(str).str.extract('(\d+)').fillna(-1)
    
    embarked_dummies = pd.get_dummies(df['embarked'], prefix='embarked')
    
    keep = ['age', 'cabin', 'fare', 'sex', 'pclass', 'sibsp', 'ticket']
    
    out = pd.concat([df[keep], embarked_dummies],
                   axis=1, sort=False)
    
    for col in out:
        out[col] = out[col].astype('float')
    
    return out