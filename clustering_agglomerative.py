from sklearn.cluster import AgglomerativeClustering
from linkage import Linkage

def agglo(df_small, small, linkage):
    print("RUNNING AGGLOMERATIVE CLUSTERING")

    match linkage:
        case linkage.COMPLETE:
            model = AgglomerativeClustering(
                n_clusters=100,
                linkage='complete' 
            )
        case linkage.AVERAGE:
            model = AgglomerativeClustering(
                n_clusters=100,
                linkage='average' 
            )
        case linkage.SINGLE:
            model = AgglomerativeClustering(
                n_clusters=100,
                linkage='single' 
            )

    labels_small = model.fit_predict(small)
    df_small["cluster"] = labels_small

    return df_small