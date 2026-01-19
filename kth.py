import pandas as pd
from text_mining import top_word
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from methode import Methode
from linkage import Linkage
import clustering_kmeans
import clustering_agglomerative
import clustering_dbscan
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pydeck as pdk

df = pd.read_csv(
    "./data_clean.csv"
)
df["date_taken"] = pd.to_datetime(df["date_taken"], errors="coerce")
df_sc = df[["lat","long"]]

data_df = pd.DataFrame(data=df_sc, columns=df_sc.columns)

k = 3  
X = data_df.sample(5000, random_state=42).to_numpy()

nn = NearestNeighbors(n_neighbors=k)
nn.fit(X)

distances, _ = nn.kneighbors(X)
k_distances = np.sort(distances[:, k-1])

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(k_distances)
ax.set_title(f"k-distance plot (k = {k})")
ax.set_xlabel("Points sorted")
ax.set_ylabel("Distance to k-th nearest neighbor")

st.pyplot(fig)
