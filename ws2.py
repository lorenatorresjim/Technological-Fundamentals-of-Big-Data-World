import requests
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

url = "https://api.idescat.cat/onomastica/v1/nadons.json"
url_comarques = "https://api.idescat.cat/emex/v1/nodes.json?geo=cat&tipus=com"
years_to_get = list(range(2020, 2025))  # From 2020 t0 2024
all_records = []

print(f"Parallel download of data from: {years_to_get}")

def fetch_page(year, start):
    """Descarga una página de nombres desde IDESCAT."""
    params = {
        "lang": "en",
        "orderby": "v",
        "desc": "1",
        "posicio": str(start),
        "t": str(year)   # Parameter of each year
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        feed = data.get("feed", {})
        entries = feed.get("entry", [])
        return entries
    except Exception as e:
        print(f"Error fetching year {year}, start {start}: {e}")
        return []

# Parallel download per year
for year in years_to_get:
    print(f"\nYear {year}: downloading names...")
    starts = list(range(0, 1500, 25))
    entries = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch_page, year, s) for s in starts]
        for f in as_completed(futures):
            result = f.result()
            if result:
                entries.extend(result)

    print(f"{len(entries)} names downloaded for {year}")

    for e in entries:
        name = e.get("title", "").strip().upper()
        ono_f = e.get("ono:f", {})
        info = ono_f.get("ono:c", {})
        pos1 = ono_f.get("ono:pos1", {})
        births = pos1.get("ono:v", "0")
        sex = info.get("sex", "")
        try:
            births = int(births)
        except:
            births = 0

        all_records.append({
            "year": int(year),
            "name": name,
            "sex": sex,
            "births": births
        })

# Dataframe created
df = pd.DataFrame(all_records)
df.drop_duplicates(subset=["year", "name", "sex"], inplace=True)


# Computations
total_births_5yrs = df["births"].sum()
total_women_5yrs = df[df["sex"] == "f"]["births"].sum()
total_maria_5yrs = df[df["name"].isin(["MARIA", "MARÍA"])]["births"].sum()
total_maria_women_5yrs = df[(df["sex"] == "f") & (df["name"].isin(["MARIA", "MARÍA"]))]["births"].sum()

frequency = total_maria_women_5yrs / total_women_5yrs if total_women_5yrs else 0
maria_per_thousand_total = (total_maria_5yrs / total_births_5yrs) * 1000 if total_births_5yrs else 0
maria_per_thousand_females = frequency * 1000

print("\nResults from 2020–2024")
print(f"Total newborns: {total_births_5yrs}")
print(f"Total women born: {total_women_5yrs}")
print(f"Total women named María: {total_maria_women_5yrs}")
print(f"María's frequency: {frequency:.4f}")
print(f"Marías per thousand newborns (total): {maria_per_thousand_total:.2f}")
print(f"Marías per thousand newborns (only females): {maria_per_thousand_females:.2f}")

# Table per year
df_maria_yearly = (
    df[(df["sex"] == "f") & (df["name"].isin(["MARIA", "MARÍA"]))]  # only women named María
    .groupby("year")["births"]
    .sum()
    .reset_index()
)

df_females_yearly = (
    df[df["sex"] == "f"]
    .groupby("year")["births"]
    .sum()
    .reset_index()
)

merged = pd.merge(df_maria_yearly, df_females_yearly, on="year", suffixes=("_maria", "_total_females"))
merged["percent_maria"] = (merged["births_maria"] / merged["births_total_females"]) * 100

print("\nMaría by year (absolute and % among females):")
print(merged)

# Plots
plt.figure(figsize=(8, 5))
plt.plot(merged["year"], merged["births_maria"], marker="o", color="#FF9AA2", label="Marías born")
plt.title("Evolution in the number of newborns named María in Catalonia (2020–2024)")
plt.xlabel("Year")
plt.ylabel("Births")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()




# Region of Catalonia with more Marías
print("\nLooking for the region with more Marías born (2020–2024)...")

import time 

# regions and official IDESCAT codes (geo = com:<codi>)
comarques = [
    {"codi": "01", "nom": "Alt Camp"},
    {"codi": "02", "nom": "Alt Empordà"},
    {"codi": "03", "nom": "Alt Penedès"},
    {"codi": "04", "nom": "Alt Urgell"},
    {"codi": "05", "nom": "Alta Ribagorça"},
    {"codi": "06", "nom": "Anoia"},
    {"codi": "07", "nom": "Bages"},
    {"codi": "08", "nom": "Baix Camp"},
    {"codi": "09", "nom": "Baix Ebre"},
    {"codi": "10", "nom": "Baix Empordà"},
    {"codi": "11", "nom": "Baix Llobregat"},
    {"codi": "12", "nom": "Baix Penedès"},
    {"codi": "13", "nom": "Barcelonès"},
    {"codi": "14", "nom": "Berguedà"},
    {"codi": "15", "nom": "Cerdanya"},
    {"codi": "16", "nom": "Conca de Barberà"},
    {"codi": "17", "nom": "Garraf"},
    {"codi": "18", "nom": "Garrigues"},
    {"codi": "19", "nom": "Garrotxa"},
    {"codi": "20", "nom": "Gironès"},
    {"codi": "21", "nom": "Maresme"},
    {"codi": "22", "nom": "Moianès"},
    {"codi": "23", "nom": "Montsià"},
    {"codi": "24", "nom": "Noguera"},
    {"codi": "25", "nom": "Osona"},
    {"codi": "26", "nom": "Pallars Jussà"},
    {"codi": "27", "nom": "Pallars Sobirà"},
    {"codi": "28", "nom": "Pla d'Urgell"},
    {"codi": "29", "nom": "Pla de l'Estany"},
    {"codi": "30", "nom": "Priorat"},
    {"codi": "31", "nom": "Ribera d'Ebre"},
    {"codi": "32", "nom": "Ripollès"},
    {"codi": "33", "nom": "Segarra"},
    {"codi": "34", "nom": "Segrià"},
    {"codi": "35", "nom": "Selva"},
    {"codi": "36", "nom": "Solsonès"},
    {"codi": "37", "nom": "Tarragonès"},
    {"codi": "38", "nom": "Terra Alta"},
    {"codi": "39", "nom": "Urgell"},
    {"codi": "40", "nom": "Vallès Occidental"},
    {"codi": "41", "nom": "Vallès Oriental"}
]

resultats_comarques = []

for comarca in comarques:
    total_marias = 0
    for year in years_to_get: 
        params = {
            "lang": "en",
            "geo": f"com:{comarca['codi']}",
            "t": str(year)
        }
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code != 200:
                continue

            data = r.json()
            if isinstance(data, str) or "feed" not in data:
                continue

            entries = data["feed"].get("entry", [])
            for e in entries:
                name = e.get("title", "").strip().upper()
                if name in ["MARIA", "MARÍA"]:
                    births = (
                        e.get("ono:f", {})
                         .get("ono:pos1", {})
                         .get("ono:v", "0")
                    )
                    try:
                        total_marias += int(births)
                    except ValueError:
                        pass
        except Exception:
            pass
        time.sleep(0.15)  # pause for not saturating the API

    resultats_comarques.append({
        "comarca": comarca["nom"],
        "marias_total": total_marias
    })
    print(f" {comarca['nom']}: {total_marias} Marías totales")


df_comarques = pd.DataFrame(resultats_comarques)
df_comarques = df_comarques.sort_values(by="marias_total", ascending=False)

# results
if not df_comarques.empty:
    comarca_top = df_comarques.iloc[0]
    print(f"\nThe region with more Marías born between 2020 and 2024 is: "
          f"{comarca_top['comarca']} ({comarca_top['marias_total']} births)")
else:
    print("No data could be obtained by region.")
