import pandas as pd

files = ["fx_q1.csv", "fx_q2.csv", "fx_q3.csv", "fx_q4.csv"]
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).drop_duplicates()

# garde uniquement les colonnes attendues si besoin
keep = ["Id", "Start", "Name", "Impact", "Currency"]
df = df[keep]

df.to_csv("fxstreet_calendar.csv", index=False)
print("Saved fxstreet_calendar.csv", len(df))