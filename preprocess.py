import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


_df = pd.read_csv('./datasets/parkinsons.data')

_x = _df.drop(['name'], 1)
feature_names = _x.drop(['status'], 1).columns
_x = np.array(_x.drop(['status'], 1))
_y = np.array(_df['status'])

x_train, x_test, y_train, y_test = train_test_split(_x, _y, test_size=0.33, random_state=42)
__all__ = ['feature_names', 'x_train', 'x_test', 'y_train', 'y_test']