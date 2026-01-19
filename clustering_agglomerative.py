from sklearn.cluster import AgglomerativeClustering


def agglo(df_small, small):
    print("RUNNING AGGLOMERATIVE CLUSTERING")


    model1 = AgglomerativeClustering(
        n_clusters=100,
        linkage='complete' 
    )
    model2 = AgglomerativeClustering(
        n_clusters=100,
        linkage='average' 
    )
    model3 = AgglomerativeClustering(
        n_clusters=100,
        linkage='single' 
    )

    labels_small = model1.fit_predict(small)
    df_small["cluster"] = labels_small

    return df_small