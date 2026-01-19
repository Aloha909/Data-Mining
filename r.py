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
    "./flickr_data2.csv"
)
print("nombre de données initiales :")
print(df.shape)

print(df.columns.tolist())




print("Duplicats id:", df["id"].duplicated().sum())

print("lat min/max:", df[" lat"].min(), df[" lat"].max())
print("lon min/max:", df[" long"].min(), df[" long"].max())




df.columns = df.columns.str.strip()


df["non_na_count"] = df.notna().sum(axis=1)
df = df.sort_values(by=["id", "non_na_count"], ascending=[True, False])
df = df.drop_duplicates(subset=["id"], keep="first")
df = df.drop(columns=["non_na_count"])

print("Après suppression des id duplicate :")
print(df.shape)

cols_unnamed = df.columns[df.columns.str.startswith("Unnamed")]
df = df.loc[~df[cols_unnamed].notna().any(axis=1)]
##print(df.loc[df[cols_unnamed].notna().any(axis=1), cols_unnamed])

##print(df.loc[df["Unnamed: 18"].notna(), ("Unnamed: 18")])

df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
df["long"] = pd.to_numeric(df["long"], errors="coerce")
df["date_taken_minute"] = pd.to_numeric(df["date_taken_minute"], errors="coerce")
df["date_taken_hour"] = pd.to_numeric(df["date_taken_hour"], errors="coerce")
df["date_taken_day"] = pd.to_numeric(df["date_taken_day"], errors="coerce")
df["date_taken_month"] = pd.to_numeric(df["date_taken_month"], errors="coerce")
df["date_taken_year"] = pd.to_numeric(df["date_taken_year"], errors="coerce")


cols = [
    "date_taken_year",
    "date_taken_month",
    "date_taken_day",
    "date_taken_hour",
    "date_taken_minute"
]

df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")

df["date_taken"] = pd.to_datetime(
    dict(
        year=df["date_taken_year"],
        month=df["date_taken_month"],
        day=df["date_taken_day"],
        hour=df["date_taken_hour"],
        minute=df["date_taken_minute"]
    ),
    errors="coerce"
)


df = df.drop(columns=["date_taken_minute"])
df = df.drop(columns=["date_taken_hour"])
df = df.drop(columns=["date_taken_day"])
df = df.drop(columns=["date_taken_month"])
df = df.drop(columns=["date_taken_year"])


df = df.drop(columns=["date_upload_minute"])
df = df.drop(columns=["date_upload_hour"])
df = df.drop(columns=["date_upload_day"])
df = df.drop(columns=["date_upload_month"])
df = df.drop(columns=["date_upload_year"])

for col in ["tags", "title", "user"]:
    if col in df.columns:
        df[col] = df[col].astype("string").fillna("")


##print(df[["lat", "long"]].isna().sum())
print("après suppression des colonnes upload et regroupement de taken dans un dataframe + suppression des lignes qui ont des valeurs dans les 3 dernières colonnes :")
print(df.shape)

df_sc = df[["lat","long"]]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_sc)

print(scaled_data)

scaled_data_df = pd.DataFrame(data=scaled_data, columns=df_sc.columns)
scaled_data_df.head()

small = scaled_data_df.sample(5000, random_state=9)  



k = 100

kmeans = KMeans(n_clusters=k, init='k-means++')

kmeans.fit(scaled_data_df)

df["cluster"] = (kmeans.labels_)

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

labels_small_agg = model3.fit_predict(small)
#plt.figure()
km = []
l = []

# for i in range(2, 100):
#     kmeans = KMeans(n_clusters=i*2, init='k-means++')
#     kmeans.fit(scaled_data_df)
#     km.append(kmeans.inertia_)
#     l.append(i*2)

# plt.plot(l, km, marker="o")
# plt.xlabel("Number of clusters (k)")
# plt.ylabel("Inertia")
# plt.title("Elbow method")
# #plt.show()

# st.pyplot(plt)

#print("salut")

# DBSCAN


X = small.to_numpy()  

eps_values = np.arange(0.0001, 0.002, 0.00005)
min_samples_values = range(1,2)


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
# results_df = results_df[ (results_df["n_clusters"] >= 80)]

# results_df.head(10)

# if len(results_df) > 0:
#         best = results_df.iloc[0]
#         best_eps = float(best["eps"])
#         best_ms = int(best["min_samples"])

#         st.write("Best params:", best_eps, best_ms)

#         labels_best = DBSCAN(eps=best_eps, min_samples=best_ms).fit_predict(X)
#         st.write("Clusters (hors bruit):", len(set(labels_best)) - (1 if -1 in labels_best else 0))
#         st.write("Noise points:", int((labels_best == -1).sum()))

# #0.45000000000000007 8
# # 0.15000000000000002 4
# labels = DBSCAN(eps=best_eps, min_samples=best_ms).fit_predict(X)

labels = DBSCAN(eps=0.0160000, min_samples=2).fit_predict(X)
# cat_df = pd.DataFrame()

# for col in ["lat", "long"]:
#     cat_df[col] = pd.qcut(
#         df[col],
#         q=100
#         )
#cat_df




#binary_df = pd.get_dummies(cat_df)






df_map = df[["lat", "long", "cluster"]].dropna().copy()

df_small = df.iloc[small.index].copy()
#df_small["cluster"] = labels_small_agg
df_small["cluster"] = labels

palette = []
for i in range(100):
    r = (50 + i * 40) % 256
    g = (100 + i * 70) % 256
    b = (150 + i * 90) % 256
    palette.append([r, g, b])

# palette = [
#     [255, 0, 0],    
#     [0, 0, 255],     
#     [0, 200, 0],     
#     [160, 32, 240], 
#     [255, 165, 0],   
#     [0, 255, 255],   
#     [255, 0, 255],   
#     [128, 128, 128], 
#     [255, 255, 0],   
#     [0, 128, 128]    
# ]



# df_map["color"] = df_map["cluster"].apply(
#     lambda c: palette[int(c) % len(palette)] if int(c) >= 0 else [0, 0, 0]
# )

# df_small["color"] = df_small["cluster"].apply(
#     lambda c: palette[int(c) % len(palette)] if int(c) >= 0 else [0, 0, 0]
# )



# layer = pdk.Layer(
#     "ScatterplotLayer",
#     data=df_small,
#     get_position="[long, lat]",
#     get_fill_color="color",
#     get_radius=25,          
#     pickable=True,
#     opacity=0.7
# )

# view_state = pdk.ViewState(
#     latitude=float(df_small["lat"].mean()),
#     longitude=float(df_small["long"].mean()),
#     zoom=12
# )

# st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))

photos_per_month = (
    df_small
    .groupby(df["date_taken"].dt.to_period("M"))
    .size()
    .reset_index(name="count")
)

photos_per_month["date_taken"] = photos_per_month["date_taken"].dt.to_timestamp()

fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(
    photos_per_month["date_taken"],
    photos_per_month["count"],
    linewidth=2
)

ax.set_title("Number of photos over time")
ax.set_xlabel("Date")
ax.set_ylabel("Number of photos")

plt.tight_layout()
st.pyplot(fig)


# # bornes
# min_d = df_small["date_taken"].min().date()
# max_d = df_small["date_taken"].max().date()

# start, end = st.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)

# df_t = df_small[(df_small["date_taken"].dt.date >= start) & (df_small["date_taken"].dt.date <= end)]

# st.write("Points in range:", len(df_t))

# # échantillon si trop gros
# max_points = 50000
# if len(df_t) > max_points:
#     df_t = df_t.sample(max_points, random_state=42)

# layer = pdk.Layer(
#     "ScatterplotLayer",
#     data=df_t,
#     get_position="[long, lat]",
#     get_radius=25,
#     opacity=0.6,
#     pickable=True
# )

# view_state = pdk.ViewState(
#     latitude=float(df_t["lat"].mean()),
#     longitude=float(df_t["long"].mean()),
#     zoom=12
# )

# st.pydeck_chart(pdk.Deck(
#     layers=[layer],
#     initial_view_state=view_state,
#     tooltip={"text": "date: {date_taken}\nid: {id}"}
# ))



# # layer = pdk.Layer(
# #     "ScatterplotLayer",
# #     data=df_map,
# #     get_position="[long, lat]",
# #     get_fill_color="color",
# #     get_radius=25,          
# #     pickable=True,
# #     opacity=0.7
# # )

# # view_state = pdk.ViewState(
# #     latitude=float(df_map["lat"].mean()),
# #     longitude=float(df_map["long"].mean()),
# #     zoom=12
# # )

# #st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))