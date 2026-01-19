import pandas as pd
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler

def kmeans(df, scaled_data_df, k):
    print("RUNNING KMEANS")

    if (len(scaled_data_df)>k):
        kmeans = KMeans(n_clusters=k, init='k-means++')
        kmeans.fit(scaled_data_df)
        df["cluster"] = (kmeans.labels_)
        df_map = df[["lat", "long", "tags", "cluster"]].dropna().copy()
        return df_map
    else:
        return None