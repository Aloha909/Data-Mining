import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium



df = pd.read_csv(
    "./flickr_data2.csv"
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




n = st.slider("Number of points displayed", 500, 5000, 2000, 500)
sample = df.sample(n, random_state=42)

m = folium.Map(
    location=[sample["lat"].mean(), sample["long"].mean()],
    zoom_start=12
)

mc = MarkerCluster().add_to(m)

for _, r in sample.iterrows():
    folium.Marker(
        [r["lat"], r["long"]],
        popup=f"<a href=\"https://www.flickr.com/photos/{r['user']}/{r['id']}\">{r['title']}</a>"
    ).add_to(mc)

st_folium(m, width=1000, height=600)