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


df = pd.read_csv("./data_clean.csv")

k = 3


X_deg = df[["lat", "long"]].sample(5000, random_state=42).to_numpy()

# degrés -> radians 
X_rad = np.radians(X_deg)

nn = NearestNeighbors(n_neighbors=k, metric="haversine")
nn.fit(X_rad)

distances, _ = nn.kneighbors(X_rad)

k_dist_rad = np.sort(distances[:, k - 1])

# radians -> mètres
EARTH_RADIUS_M = 6371000.0
k_dist_m = k_dist_rad * EARTH_RADIUS_M

# Plot
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(k_dist_m)


ax.set_ylim(0, 200)

ax.set_title(f"k-distance plot (k = {k})")
ax.set_xlabel("Points sorted")
ax.set_ylabel("Distance to k-th nearest neighbor (meters)")

st.pyplot(fig)
