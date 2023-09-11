from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

df_rating = pd.read_csv("Data/ratings.csv")

def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))

