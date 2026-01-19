import clustering_kmeans
from enum import Enum

import streamlit as st
import pydeck as pdk

class Methode(Enum):
    KMEANS = 1,

m = Methode.KMEANS

match m:
    case Methode.KMEANS:
        df_map = clustering_kmeans.kmeans()
    case _:
        print("erreur")
        



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
