import pandas as pd

m1 = pd.read_csv("EURUSD_M1.csv", sep=r"\s+", engine="python")
m1.columns = [c.strip("<>").lower() for c in m1.columns]
m1["dt"] = pd.to_datetime(m1["date"] + " " + m1["time"], format="%Y.%m.%d %H:%M:%S", utc=True)

iso = m1["dt"].dt.isocalendar()
m1["iso_year"] = iso.year.astype(int)
m1["iso_week"] = iso.week.astype(int)
m1["weekday"] = m1["dt"].dt.weekday
m1["hhmm"] = m1["dt"].dt.strftime("%H:%M")

unsafe = pd.read_csv("unsafe_weeks.csv")
unsafe["is_unsafe_week"] = 1

m1 = m1.merge(unsafe, on=["iso_year", "iso_week"], how="left")
m1["is_safe_week"] = (m1["is_unsafe_week"].fillna(0) == 0).astype(int)
m1 = m1.drop(columns=["is_unsafe_week"])

m1.to_parquet("EURUSD_M1_with_safeweek.parquet", index=False)
print(m1[["dt","iso_year","iso_week","is_safe_week"]].head())