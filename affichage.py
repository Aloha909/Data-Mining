import pandas as pd
from text_mining import top_words
from sklearn.preprocessing import StandardScaler
from methode import Methode
from linkage import Linkage
import clustering_kmeans
import clustering_agglomerative
import clustering_dbscan

import streamlit as st
import pydeck as pdk

m = Methode.KMEANS
l = Linkage.COMPLETE
k = 300

st.sidebar.header("Paramètres")
method_label = st.sidebar.selectbox(
    "Méthode de clustering",
    ("Agglomerative", "K-Means", "DBSCAN")
)

text_mining_method_label = st.sidebar.selectbox(
    "Méthode de text mining",
    ("Lemmatisation + TF-IDF", "NER + log-odds")
)

if method_label == "K-Means":
    m = Methode.KMEANS
    k = st.sidebar.slider("Nombre de clusters (k)", 1, 1000, 100, 10)
    data = st.sidebar.selectbox("Taille du dataset : ", ("Reduit", "Entier"))
elif method_label == "Agglomerative":
    linkage_str = st.sidebar.selectbox("Type de linkage", ("Complete", "Average", "Single"))
    match linkage_str:
        case "Complete":
            l = Linkage.COMPLETE
        case "Average":
            l = Linkage.AVERAGE
        case "Single":
            l = Linkage.SINGLE
    nb_clust_agglo = st.sidebar.slider("Nombre de clusters", 1, 1000, 100, 10)
    m = Methode.AGGLO
else:
    dist_metre = st.sidebar.slider("Distance minimale en mètres", 1,100,5)
    eps = dist_metre / 6384415.0
    min_sample = st.sidebar.slider("Minimum sample", 1, 20, 2)
    m = Methode.DBSCAN
    
calcul_tags = st.toggle('Calculer les tags', value=True)

df = pd.read_csv(
    "./data_clean.csv"
)
df["date_taken"] = pd.to_datetime(df["date_taken"], errors="coerce")

min_year = int(df["date_taken"].dt.year.min())
max_year = int(df["date_taken"].dt.year.max())

start_year, end_year = st.slider(
    "Années de prise de vue",
    min_value=min_year,
    max_value=max_year,
    value=(2005, 2010),
    step=1
)

df = df[
    (df["date_taken"].dt.year >= start_year) &
    (df["date_taken"].dt.year <= end_year)
]

df_sc = df[["lat","long"]]

data_df = pd.DataFrame(data=df_sc, columns=df_sc.columns)
data_df.head()

n = min(5000, len(data_df))
small = data_df.sample(n, random_state=9)
df_small = df.loc[small.index].copy()

match m:
    case Methode.KMEANS:
        if data == "Reduit":
            df_map = clustering_kmeans.kmeans(df_small, small, k)
        elif data == "Entier":
            df_map = clustering_kmeans.kmeans(df, data_df, k)
    case Methode.AGGLO:
        df_map = clustering_agglomerative.agglo(df_small, small, l, nb_clust_agglo)
    case Methode.DBSCAN:
        df_map = clustering_dbscan.dbscan(df_small, small, eps, min_sample)
    case _:
        print("erreur")

if (not(df_map.empty) and calcul_tags):
    cluster_tags = top_words(df_map, text_mining_method_label, 3)

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

if calcul_tags:
    df_map["top_tags"] = df_map["cluster"].apply(
        lambda c: ", ".join(cluster_tags[c]) if c in cluster_tags else ""
    )

    print("## DEBUG: tags calculés ##")
    print(df_map[["cluster", "top_tags"]].drop_duplicates().head())

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

if calcul_tags:
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{top_tags}"}))
else:
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))
