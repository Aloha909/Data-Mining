import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN



def dbscan(df_small, small, eps, min_sample):
    print("RUNNING DBSCAN")
    X = np.radians(small.to_numpy())
    labels = DBSCAN(eps=eps, min_samples=min_sample, metric='haversine', algorithm='ball_tree').fit_predict(X)
    df_small["cluster"] = labels

    return df_small
