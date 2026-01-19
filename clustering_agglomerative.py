from sklearn.cluster import AgglomerativeClustering
from linkage import Linkage

def agglo(df_small, small, linkage, nb):
    print("RUNNING AGGLOMERATIVE CLUSTERING")

    match linkage:
        case linkage.COMPLETE:
            model = AgglomerativeClustering(
                n_clusters=nb,
                linkage='complete' 
            )
        case linkage.AVERAGE:
            model = AgglomerativeClustering(
                n_clusters=nb,
                linkage='average' 
            )
        case linkage.SINGLE:
            model = AgglomerativeClustering(
                n_clusters=nb,
                linkage='single' 
            )

    labels_small = model.fit_predict(small)
    df_small["cluster"] = labels_small

    return df_small