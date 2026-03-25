import pandas as pd

CSV = "data/hospitals_demo_valid_coords.csv"
LAT, LON = "Latitude", "Longitude"

df = pd.read_csv(CSV)
df[LAT] = pd.to_numeric(df[LAT], errors="coerce")
df[LON] = pd.to_numeric(df[LON], errors="coerce")
df = df.dropna(subset=[LAT, LON])

print("Total rows with coords:", len(df))
print("Unique coordinate pairs:", df[[LAT, LON]].drop_duplicates().shape[0])

# show the most common coordinate pair(s)
top = (
    df.groupby([LAT, LON])
      .size()
      .sort_values(ascending=False)
      .head(10)
)
print("\nTop repeated coordinates (lat, lon -> count):")
print(top)
