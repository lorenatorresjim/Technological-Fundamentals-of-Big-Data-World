import json
import pandas as pd
import matplotlib.pyplot as plt
import requests

url = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/ei_bsco_m?format=JSON&unit=BAL&indic=BS-CSMCI&s_adj=SA&lang=EN"
response = requests.get(url)
data = response.json()

values = data["value"]
dims = data["dimension"]
geo_labels = dims["geo"]["category"]["label"]
time_labels = dims["time"]["category"]["label"]
geo_index = dims["geo"]["category"]["index"]
time_index = dims["time"]["category"]["index"]
geo_codes = list(geo_index.keys())
time_codes = list(time_index.keys())

records = []
n_time = len(time_codes) # Number of months
for key, val in values.items():
    k = int(key)
    geo_pos = k // n_time
    time_pos = k % n_time
    geo = geo_codes[geo_pos]
    time = time_codes[time_pos]
    records.append((geo, time_labels[time], val))

# Create DataFrame
df = pd.DataFrame(records, columns=["geo", "time", "value"])
df["time"] = pd.to_datetime(df["time"], format="%Y-%m")
df_eu = df[df["geo"] == "EU27_2020"]
df_es = df[df["geo"] == "ES"]

#Last 5 years from 2020
df_eu_last5 = df_eu[df_eu["time"] >= "2020-01-01"]
avg_eu_per_year = df_eu_last5.groupby(df_eu_last5["time"].dt.year)["value"].mean()
df_eu_last5 = df_eu[df_eu["time"] >= "2020-01-01"]
avg_eu_last5 = df_eu_last5["value"].mean()
print("\n🇪🇺 Average EU consumer confidence per year (last 5 years):")
print(avg_eu_per_year)
print("Average EU consumer confidence (last 5 years):", avg_eu_last5)

# UE 2024
df_eu_2024 = df_eu[df_eu["time"].dt.year == 2024]
avg_eu_2024 = df_eu_2024.groupby(df_eu_2024["time"].dt.month)["value"].mean()
print("\n🇪🇺 EU consumer confidence per month (2024):")
print(avg_eu_2024)

# Spain 2024
df_es_2024 = df_es[df_es["time"].dt.year == 2024]
avg_es_2024 = df_es_2024.groupby(df_es_2024["time"].dt.month)["value"].mean()
print("\n🇪🇸 Spanish consumer confidence per month (2024):")
print(avg_es_2024)

#Plot last 5 years from 2020
plt.figure(figsize=(7, 5))
plt.bar(avg_eu_per_year.index.astype(str), avg_eu_per_year.values, color="steelblue")
plt.title("Average EU Consumer Confidence – Last 5 Years")
plt.xlabel("Year")
plt.ylabel("Confidence Index (Balance)")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()


# Plot UE 2024
plt.figure(figsize=(8, 5))
plt.bar(avg_eu_2024.index, avg_eu_2024.values, color="skyblue")
plt.xticks(range(1, 13),
           ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
plt.title("EU Consumer Confidence 2024 (Monthly Average)")
plt.xlabel("Month")
plt.ylabel("Confidence Index (Balance)")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


# Plot Spain 2024
plt.figure(figsize=(8, 5))
plt.bar(avg_es_2024.index, avg_es_2024.values, color="darkorange")
plt.xticks(range(1, 13),
            ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
plt.title("Spain Consumer Confidence 2024 (Monthly Average)")
plt.xlabel("Month")
plt.ylabel("Confidence Index (Balance)")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

