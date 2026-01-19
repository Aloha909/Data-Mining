import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("TkAgg")

import streamlit as st
import pydeck as pdk

import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score




df = pd.read_csv(
    "./data_clean.csv"
)

small = df.sample(15000, random_state=9)
df_small = df.iloc[small.index].copy()

df_small["date_taken"] = pd.to_datetime(df_small["date_taken"], errors="coerce")


# photos_per_month = (
#     df_small
#     .groupby(df_small["date_taken"].dt.to_period("M"))
#     .size()
#     .reset_index(name="count")
# )

# photos_per_month["date_taken"] = photos_per_month["date_taken"].dt.to_timestamp()

# fig, ax = plt.subplots(figsize=(10, 4))

# ax.plot(
#     photos_per_month["date_taken"],
#     photos_per_month["count"],
#     linewidth=2
# )

# ax.set_title("Nombre de photos au cours du temps")
# ax.set_xlabel("Date")
# ax.set_ylabel("Nombre de photos")


# ax.set_xlim(left=pd.Timestamp("2000-01-01"))

# plt.tight_layout()
# st.pyplot(fig)


min_year = int(df_small["date_taken"].dt.year.min())
max_year = int(df_small["date_taken"].dt.year.max())

start_year, end_year = st.slider(
    "Year range",
    min_value=min_year,
    max_value=max_year,
    value=(2005, 2010),
    step=1
)

df_t = df_small[
    (df_small["date_taken"].dt.year >= start_year) &
    (df_small["date_taken"].dt.year <= end_year)
]


X = df_t[["lat","long"]]

labels = KMeans(n_clusters=300).fit_predict(X)

df_t["cluster"] = labels


palette = [
    [255, 0, 0],    
    [0, 0, 255],     
    [0, 200, 0],     
    [160, 32, 240], 
    [255, 165, 0],   
    [0, 255, 255],   
    [255, 0, 255],   
    [128, 128, 128], 
    [255, 255, 0],   
    [0, 128, 128]    
]

df_t["color"] = df_t["cluster"].apply(
    lambda c: palette[int(c) % len(palette)] if int(c) >= 0 else [0, 0, 0]
)

layer = pdk.Layer(
    "ScatterplotLayer",
    data=df_t,
    get_position="[long, lat]",
    get_fill_color="color",
    get_radius=25,          
    pickable=True,
    opacity=0.7
)

view_state = pdk.ViewState(
    latitude=float(df_t["lat"].mean()),
    longitude=float(df_t["long"].mean()),
    zoom=12
)

st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))
