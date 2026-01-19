


import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


def dbscan(df_small, small, eps, min_sample):
    print("RUNNING DBSCAN")

    X = small.to_numpy()  

    # eps_values = np.arange(0.1, 0.81, 0.05)   
    # min_samples_values = range(3, 9)

    # results = []

    # for eps in eps_values:
    #     for ms in min_samples_values:
    #         labels = DBSCAN(eps=eps, min_samples=ms).fit_predict(X)

    #         # enlever le bruit
    #         mask = labels != -1

    #         # si trop peu de points gardés, ou un seul cluster => pas de silhouette possible
    #         if mask.sum() < 10:
    #             continue

    #         n_clusters = len(set(labels[mask]))
    #         if n_clusters < 2:
    #             continue

    #         sil = silhouette_score(X[mask], labels[mask])

    #         n_noise = (labels == -1).sum()

    #         results.append([eps, ms, sil, n_clusters, n_noise, n_noise / len(labels)])

    # results_df = pd.DataFrame(
    #     results,
    #     columns=["eps", "min_samples", "silhouette", "n_clusters", "n_noise", "noise_ratio"]
    # ).sort_values("silhouette", ascending=False)
    # results_df = results_df[ (results_df["n_clusters"] >= 50)]

    # results_df.head(10)

    # if len(results_df) > 0:
    #         best = results_df.iloc[0]
    #         best_eps = float(best["eps"])
    #         print(best_eps)
    #         best_ms = int(best["min_samples"])

    labels = DBSCAN(eps=eps, min_samples=min_sample).fit_predict(X)
    df_small["cluster"] = labels


    return df_small
