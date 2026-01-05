
import pydeck as pdk
import streamlit as st
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.discriminant_analysis import StandardScaler


df = pd.read_csv(
    "./data_clean.csv"
)
df_sc = df[["lat","long"]]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_sc)

print(scaled_data)

scaled_data_df = pd.DataFrame(data=scaled_data, columns=df_sc.columns)
scaled_data_df.head()

k = 100

kmeans = KMeans(n_clusters=k, init='k-means++')

kmeans.fit(scaled_data_df)

df["cluster"] = (kmeans.labels_)

df_map = df[["lat", "long", "cluster"]].dropna().copy()

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

df_map["color"] = df_map["cluster"].apply(
    lambda c: palette[int(c) % len(palette)] if int(c) >= 0 else [0, 0, 0]
)

layer = pdk.Layer(
    "ScatterplotLayer",
    data=df_map,
    get_position="[long, lat]",
    get_fill_color="color",
    get_radius=25,          
    pickable=True,
    opacity=0.7
)

view_state = pdk.ViewState(
    latitude=float(df_map["lat"].mean()),
    longitude=float(df_map["long"].mean()),
    zoom=12
)

st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))
