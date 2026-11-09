# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 22:14:52 2026

@author: Admin
"""

# ======================================
# LOAD DISTRICT SHAPEFILE
# ======================================

di = gpd.read_file(
    "D:/Masters Report/shape_file/gadm41_BGD_shp/gadm41_BGD_1.shp")

district_name_field = "NAME_1"   # change if your column name differs
districts = districts.to_crs("EPSG:4326")

# ======================================
# LOAD ERA5 DATA
# ======================================

temp_files = [
    "D:/Masters Report/Report data/Temp-dewpoint-surfacepressure(1991-1995).nc",
    "D:/Masters Report/Report data/Temp-dewpoint-surfacepressure(1996-2005).nc",
    "D:/Masters Report/Report data/Temp-dewpoint-surfacepressure(2006-2015).nc",
    "D:/Masters Report/Report data/Temp-dewpoint-surfacepressure(2016-2025).nc"
]

rain_files = [
    "D:/Masters Report/Report data/precipitation-cloudcover(1991-2007)-accum.nc",
    "D:/Masters Report/Report data/precipitation-cloudcover(1991-2007)-instant.nc",
    "D:/Masters Report/Report data/Precipitation-cloudcover(2008-2025)-accum.nc",
    "D:/Masters Report/Report data/Precipitation-cloudcover(2008-2025)-instant.nc"
]

wind_files = [
    "D:/Masters Report/Report data/Wind-u-v-component(1991-2007).nc",
    "D:/Masters Report/Report data/Wind-u-v-component(2008-2025).nc"
]

temp_ds = xr.concat([xr.open_dataset(f) for f in temp_files], dim="valid_time")
rain_ds = xr.concat([xr.open_dataset(f) for f in rain_files], dim="valid_time")
wind_ds = xr.concat([xr.open_dataset(f) for f in wind_files], dim="valid_time")

# ======================================
# TEMPERATURE CONVERSION
# ======================================

temp = temp_ds['t2m'] - 273.15
dew = temp_ds['d2m'] - 273.15

hour = temp_ds['valid_time'].dt.hour

# ======================================
# DAY & NIGHT TEMPERATURE
# ======================================

day_mask = (hour >= 6) & (hour < 18)
night_mask = (hour >= 18) | (hour < 6)

day_temp = temp.where(day_mask).groupby("valid_time.date").max()
night_temp = temp.where(night_mask).groupby("valid_time.date").max()

# ======================================
# RELATIVE HUMIDITY
# ======================================

es = 6.112 * np.exp((17.67 * temp) / (temp + 243.5))
e = 6.112 * np.exp((17.67 * dew) / (dew + 243.5))

rh = (e / es) * 100
rh_daily = rh.groupby("valid_time.date").max()

# ======================================
# SPECIFIC HUMIDITY
# ======================================

sp = temp_ds['sp'] / 100
q = (0.622 * e) / (sp - 0.378 * e)
q_daily = q.groupby("valid_time.date").max()

# ======================================
# WIND SPEED
# ======================================

wind_speed = np.sqrt(wind_ds['u10']**2 + wind_ds['v10']**2)
wind_daily = wind_speed.groupby("valid_time.date").max()

# ======================================
# PRECIPITATION
# ======================================

precip = rain_ds['tp'] * 1000
precip_daily = precip.groupby("valid_time.date").sum()

# ======================================
# CLOUD COVER
# ======================================

cloud_daily = rain_ds['tcc'].groupby("valid_time.date").max()

# ======================================
# MERGE VARIABLES
# ======================================

ds = xr.Dataset({
    "day_temp": day_temp,
    "night_temp": night_temp,
    "relative_humidity": rh_daily,
    "specific_humidity": q_daily,
    "wind_speed": wind_daily,
    "precipitation": precip_daily,
    "cloud_cover": cloud_daily
})

# ======================================
# SET CRS FOR ERA5 DATA
# ======================================

ds = ds.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
ds = ds.rio.write_crs("EPSG:4326")

# ======================================
# DISTRICT EXTRACTION
# ======================================

results = []

for idx, row in districts.iterrows():

    district_geom = row.geometry
    district_name = row[district_name_field]

    clipped = ds.rio.clip([district_geom], districts.crs, drop=True)

    district_mean = clipped.mean(dim=["latitude", "longitude"])

    df = district_mean.to_dataframe().reset_index()

    df["District"] = district_name

    results.append(df)

final_df = pd.concat(results)

# ======================================
# SAVE TO EXCEL
# ======================================

final_df.to_excel(
    "D:/Masters Report/Report data/Division_Climate_Data.xlsx",
    index=False
)

print("District climate dataset created successfully.")


# ==========================
# LOAD EXCEL FILE
# ==========================

df = pd.read_excel("D:/Masters Report/Report data/Division_Climate_Data.xlsx")

df["date"] = pd.to_datetime(df["date"])

# ==========================
# CALCULATE 95th PERCENTILE
# ==========================

thresholds = df.groupby("division")["day_temp"].quantile(0.95)

df["threshold"] = df["division"].map(thresholds)

# ==========================
# IDENTIFY HOT DAYS
# ==========================

df["hot_day"] = df["day_temp"] >= df["threshold"]

# ==========================
# FUNCTION TO DETECT EVENTS
# ==========================


def heatwave_stats(group):

    group = group.sort_values("date")

    hot = group["hot_day"].astype(int)

    events = []
    length = 0

    for i in hot:
        if i == 1:
            length += 1
        else:
            if length >= 3:
                events.append(length)
            length = 0

    if length >= 3:
        events.append(length)

    heatwave_days = sum(events)
    heatwave_events = len(events)

    avg_duration = np.mean(events) if events else 0

    intensity = group.loc[group["hot_day"],
                          "day_temp"].mean() - group["threshold"].iloc[0]

    frequency = heatwave_events / group["date"].dt.year.nunique()

    return pd.Series({
        "heatwave_days": heatwave_days,
        "heatwave_events": heatwave_events,
        "heatwave_duration": avg_duration,
        "heatwave_intensity": intensity,
        "heatwave_frequency": frequency
    })

# ==========================
# APPLY FOR EACH DIVISION
# ==========================


results = df.groupby("division").apply(heatwave_stats)

# ==========================
# SAVE RESULTS
# ==========================

results.to_excel(
    "D:/Masters Report/Report data/division_heatwave_statistics.xlsx")

print("Heatwave analysis complete")


# ==========================
# LOAD DATA
# ==========================

file = r"D:/Masters Report/Report data/Division_Climate_Data.xlsx"

df = pd.read_excel(file)

# ==========================
# DATE FORMAT
# ==========================

df["date"] = pd.to_datetime(df["date"])

# ==========================
# HEATWAVE SEASON (MARCH-JUNE)
# ==========================

df["month"] = df["date"].dt.month
df = df[df["month"].isin([3, 4, 5, 6])]

# ==========================
# CALCULATE 90TH PERCENTILE THRESHOLD
# ==========================

threshold_day = df.groupby("division")["day_temp"].quantile(0.90)
threshold_night = df.groupby("division")["night_temp"].quantile(0.90)

df["threshold_day"] = df["division"].map(threshold_day)
df["threshold_night"] = df["division"].map(threshold_night)

# ==========================
# HEATWAVE DAY DETECTION
# ==========================

df["day_hw"] = df["day_temp"] >= df["threshold_day"]
df["night_hw"] = df["night_temp"] >= df["threshold_night"]

# compound heatwave
df["compound_hw"] = df["day_hw"] & df["night_hw"]

# ==========================
# DETECT ≥3 CONSECUTIVE DAYS
# ==========================

df = df.sort_values(["division", "date"])

df["hw_group"] = (
    df["day_hw"]
    .ne(df.groupby("division")["day_hw"].shift())
    .cumsum()
)

events = df[df["day_hw"]].groupby(["division", "hw_group"]).size()

events = events[events >= 3]

valid_groups = events.index

df["heatwave_event"] = df.set_index(
    ["division", "hw_group"]).index.isin(valid_groups)

# ==========================
# HEATWAVE INTENSITY
# ==========================

df["heatwave_intensity"] = df["day_temp"] - df["threshold_day"]

df.loc[~df["heatwave_event"], "heatwave_intensity"] = 0

# ==========================
# HEATWAVE STATISTICS
# ==========================

stats = []

for div in df["division"].unique():

    sub = df[df["division"] == div]

    heatwave_days = sub["heatwave_event"].sum()

    events = sub[sub["heatwave_event"]].groupby("hw_group").size()

    num_events = len(events)

    avg_duration = events.mean() if num_events > 0 else 0

    intensity = sub["heatwave_intensity"].mean()

    frequency = num_events / sub["date"].dt.year.nunique()

    stats.append([
        div,
        heatwave_days,
        num_events,
        avg_duration,
        intensity,
        frequency
    ])

stats_df = pd.DataFrame(stats, columns=[
    "division",
    "heatwave_days",
    "heatwave_events",
    "avg_duration",
    "heatwave_intensity",
    "heatwave_frequency_per_year"
])

# ==========================
# FINAL DATASET
# ==========================

final_df = df[[
    "date",
    "division",
    "day_temp",
    "night_temp",
    "day_hw",
    "night_hw",
    "compound_hw",
    "relative_humidity",
    "specific_humidity",
    "wind_speed",
    "precipitation",
    "cloud_cover",
    "heatwave_event",
    "heatwave_intensity"
]]

# ==========================
# SAVE OUTPUT
# ==========================

final_df.to_excel(
    r"D:/Masters Report/Report data/Heatwave_Final_Dataset.xlsx",
    index=False
)

stats_df.to_excel(
    r"D:/Masters Report/Report data/Heatwave_Statistics.xlsx",
    index=False
)

print("Processing complete.")


# Fig3 Statistics

df = pd.read_excel("D:/Masters Report/Report data/Heatwave_Final_Dataset.xlsx")

# ==========================
# COUNT HEATWAVE DAYS
# ==========================

summary_days = df.groupby("division").agg(
    day_hw_days=("day_hw", "sum"),
    night_hw_days=("night_hw", "sum"),
    compound_hw_days=("compound_hw", "sum")
).reset_index()

# ==========================
# COUNT EVENTS
# ==========================

events = df[df["heatwave_event"] == 1]

summary_events = events.groupby("division").agg(
    events=("heatwave_event", "count")
).reset_index()

# merge

summary = summary_days.merge(summary_events, on="division", how="left")

summary.to_excel(
    "D:/Masters Report/Report data/Heatwave_Summary.xlsx", index=False)
