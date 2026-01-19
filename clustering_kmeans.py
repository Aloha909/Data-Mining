


import pandas as pd
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler

def kmeans():
    print("RUNNING KMEANS")

    df = pd.read_csv(
        "./data_clean.csv"
    )
    df_sc = df[["lat","long"]]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_sc)

    scaled_data_df = pd.DataFrame(data=scaled_data, columns=df_sc.columns)
    scaled_data_df.head()

    k = 100

    kmeans = KMeans(n_clusters=k, init='k-means++')

    kmeans.fit(scaled_data_df)

    df["cluster"] = (kmeans.labels_)

    df_map = df[["lat", "long", "cluster"]].dropna().copy()
    return df_map
