import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


df = pd.read_csv(
    r"C:\Users\robin\OneDrive\Documents\robin\4if\Fouille\Data-Mining\flickr_data2.csv"
)
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
print(df.shape)

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




# plt.figure()
# km = []
# l = []

# for i in range(1, 50):
#     kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10)
#     kmeans.fit(scaled_data_df)
#     km.append(kmeans.inertia_)
#     l.append(i)

# plt.plot(l, km, marker="o")
# plt.xlabel("Number of clusters (k)")
# plt.ylabel("Inertia")
# plt.title("Elbow method !")
# plt.show()

##st.pyplot(plt)


# model1 = AgglomerativeClustering(
#     n_clusters=3,
#     linkage='complete' 
# )

sample = df.head(50000) 


colors = ["red", "blue", "green", "purple", "orange", "darkred", "cadetblue", "black"]

m = folium.Map(
    location=[df["lat"].mean(), df["long"].mean()],
    zoom_start=12,
    tiles="CartoDB positron"
)

for _, r in sample.iterrows():
    c = int(r["cluster"])
    color = colors[c % len(colors)]

    folium.CircleMarker(
        location=[r["lat"], r["long"]],
        radius=4,          # taille du point
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.8
    ).add_to(m)

st_folium(m, width=1000, height=600)
