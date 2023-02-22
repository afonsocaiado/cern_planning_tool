import pandas as pd
import numpy as np

from scipy.stats import zscore

import utils

q1 = pd.read_csv('..\data\processed\\q1.csv', encoding='unicode_escape')

q1 = utils.remove_nans(q1)

# remove outliers using Z-score method
q1 = q1[(np.abs(zscore(q1[['PREPARATION_DURATION', 'INSTALLATION_DURATION', 'COMMISSIONING_DURATION']])) < 3).all(axis=1)]

# what about unbalanced data?

q1.to_csv('..\data\processed\preprocessed_q1.csv', index=False)