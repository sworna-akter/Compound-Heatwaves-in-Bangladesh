# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 22:18:24 2026

@author: Admin
"""

######-----Sample Fig 2------######

# ------------------------------------------------
# 1 Load ERA5 NetCDF files
# ------------------------------------------------

files = [
    "D:/Masters Report/Report data_Final/2m temperature(1991-2010)_DayTemp.nc",
    "D:/Masters Report/Report data_Final/2m temperature(2011-2025)_DayTemp.nc",
    "D:/Masters Report/Report data_Final/2m temperature(1991-2010)_NightTemp.nc",
    "D:/Masters Report/Report data_Final/2m temperature(2011-2025)_NightTemp.nc"
]

datasets = [xr.open_dataset(f) for f in files]
ds = xr.concat(datasets, dim="valid_time")

# ------------------------------------------------
# 2 Convert Kelvin → Celsius
# ------------------------------------------------

t2m = ds["t2m"] - 273.15

# ------------------------------------------------
# 3 Fix ERA5 longitude
# ------------------------------------------------

t2m = t2m.assign_coords(
    longitude=((t2m.longitude + 180) % 360) - 180
).sortby("longitude")

# ------------------------------------------------
# 4 Set CRS
# ------------------------------------------------

t2m = t2m.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
t2m = t2m.rio.write_crs("EPSG:4326")

# ------------------------------------------------
# 5 Select JJA (MAM actually in your code)
# ------------------------------------------------

jja = t2m.where(t2m.valid_time.dt.month.isin([3, 4, 5]), drop=True)

# ------------------------------------------------
# 6 Day / Night temperature (UPDATED)
# ------------------------------------------------

# ---- Daytime (10:00–18:00) ----
day_hours = jja.where(
    (jja.valid_time.dt.hour >= 10) & (jja.valid_time.dt.hour <= 18),
    drop=True
)

# Daily max temperature (day)
day_daily = day_hours.resample(valid_time="1D").max()

# ---- Nighttime (23:00–06:00, cross midnight) ----
night_hours = jja.where(
    (jja.valid_time.dt.hour >= 23) | (jja.valid_time.dt.hour <= 6),
    drop=True
)

# Daily min temperature (night)
night_daily = night_hours.resample(valid_time="1D").min()

# ------------------------------------------------
# 7 Heatwave threshold (95th percentile)
# ------------------------------------------------

day_thresh = day_daily.quantile(0.95, dim="valid_time")
night_thresh = night_daily.quantile(0.95, dim="valid_time")

day_exceed = day_daily > day_thresh
night_exceed = night_daily > night_thresh

# ---------------------------------------------
# Apply 3 consecutive day condition
# ---------------------------------------------
def consecutive_3days(mask):
    return mask.rolling(valid_time=3, center=False).sum() >= 3

day_hw = consecutive_3days(day_exceed)
night_hw = consecutive_3days(night_exceed)
compound_hw = day_hw & night_hw

# ---------------------------------------------
# BANGLADESH-WIDE SUMMARY STATISTICS
# ---------------------------------------------

def hw_summary(mask):
    arr = mask.mean(dim=["latitude", "longitude"]).values

    hw_days = np.sum(arr)

    events = []
    length = 0

    for v in arr:
        if v == 1:
            length += 1
        else:
            if length >= 3:
                events.append(length)
            length = 0

    if length >= 3:
        events.append(length)

    num_events = len(events)

    avg_duration = np.mean(events) if num_events > 0 else 0
    max_duration = np.max(events) if num_events > 0 else 0

    return hw_days, num_events, avg_duration, max_duration


bd_day = hw_summary(day_hw)
bd_night = hw_summary(night_hw)
bd_comp = hw_summary(compound_hw)

print("\n===== BANGLADESH OVERALL SUMMARY =====")
print("Day HW  :", bd_day)
print("Night HW:", bd_night)
print("Comp HW :", bd_comp)

# ------------------------------------------------
# 8 Load Bangladesh divisions
# ------------------------------------------------

bd = gpd.read_file(
    "D:/Masters Report/shape_file/gadm41_BGD_shp/gadm41_BGD_1.shp"
).to_crs("EPSG:4326")

# ------------------------------------------------
# 9 Heatwave statistics function
# ------------------------------------------------

def hw_statistics(series):

    arr = series.astype(int)

    hw_days = arr.sum()

    events = []
    length = 0

    for v in arr:

        if v == 1:
            length += 1
        else:
            if length >= 3:
                events.append(length)
            length = 0

    if length >= 3:
        events.append(length)

    num_events = len(events)

    if num_events > 0:
        avg_duration = np.mean(events)
        max_duration = np.max(events)
    else:
        avg_duration = 0
        max_duration = 0

    return hw_days, num_events, avg_duration, max_duration


# ------------------------------------------------
# 10 Calculate statistics for each division
# ------------------------------------------------

records = []

for i, row in bd.iterrows():

    geom = [row.geometry]

    day_clip = day_hw.rio.clip(geom, bd.crs)
    night_clip = night_hw.rio.clip(geom, bd.crs)
    comp_clip = compound_hw.rio.clip(geom, bd.crs)

    day_series = day_clip.mean(dim=["latitude", "longitude"]).values
    night_series = night_clip.mean(dim=["latitude", "longitude"]).values
    comp_series = comp_clip.mean(dim=["latitude", "longitude"]).values

    d = hw_statistics(day_series)
    n = hw_statistics(night_series)
    c = hw_statistics(comp_series)

    records.append([
        row["NAME_1"],
        d, n, c
    ])

# ------------------------------------------------
# 11 Build dataframe
# ------------------------------------------------

data = []

for r in records:

    div = r[0]

    data.append([
        div,
        r[1][0], r[2][0], r[3][0],
        r[1][1], r[2][1], r[3][1],
        r[1][2], r[2][2], r[3][2],
        r[1][3], r[2][3], r[3][3]
    ])

df = pd.DataFrame(data, columns=[
    "Division",
    "Day_HWdays", "Night_HWdays", "Comp_HWdays",
    "Day_events", "Night_events", "Comp_events",
    "Day_avg", "Night_avg", "Comp_avg",
    "Day_max", "Night_max", "Comp_max"
])

# ---------------------------------------------
# SAVE FOR THESIS
# ---------------------------------------------
df.to_csv("D:/Masters Report/Report data_Final/heatwave_summary_division.csv", index=False)

print("\n===== DIVISION-WISE SUMMARY TABLE =====")
print(df)

# ---------------------------------------------
# BANGLADESH TOTAL (ALL DIVISIONS COMBINED)
# ---------------------------------------------

total_row = pd.DataFrame([{
    "Division": "Bangladesh",
    "Day_HWdays": df["Day_HWdays"].sum(),
    "Night_HWdays": df["Night_HWdays"].sum(),
    "Comp_HWdays": df["Comp_HWdays"].sum(),

    "Day_events": df["Day_events"].sum(),
    "Night_events": df["Night_events"].sum(),
    "Comp_events": df["Comp_events"].sum(),

    "Day_avg": df["Day_avg"].mean(),
    "Night_avg": df["Night_avg"].mean(),
    "Comp_avg": df["Comp_avg"].mean(),

    "Day_max": df["Day_max"].max(),
    "Night_max": df["Night_max"].max(),
    "Comp_max": df["Comp_max"].max(),
}])

df_final = pd.concat([df, total_row], ignore_index=True)

df_final.to_csv("D:/Masters Report/Report data_Final/heatwave_summary_with_bangladesh_total.csv", index=False)

# 12 Plot figure
# ------------------------------------------------

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16
})

x = np.arange(len(df))
w = 0.25

fig, ax = plt.subplots(2, 2, figsize=(16, 12))

# -----------------------------
# a Number of heatwave days
# -----------------------------

b1 = ax[0, 0].bar(x-w, df.Day_HWdays, w, color="red", label="Daytime")
b2 = ax[0, 0].bar(x, df.Night_HWdays, w, color="blue", label="Nighttime")
b3 = ax[0, 0].bar(x+w, df.Comp_HWdays, w, color="black", label="Compound")

ax[0, 0].set_title("(a) Number of Heatwave Days")
ax[0, 0].set_ylabel("#Days")

ax[0, 0].set_xticks(x)
ax[0, 0].set_xticklabels(df.Division, rotation=30)

ax[0, 0].bar_label(b1, padding=3, fontsize=14)
ax[0, 0].bar_label(b2, padding=3, fontsize=14)
ax[0, 0].bar_label(b3, padding=3, fontsize=14)

# -----------------------------
# b Number of events
# -----------------------------

b4 = ax[0, 1].bar(x-w, df.Day_events, w, color="red")
b5 = ax[0, 1].bar(x, df.Night_events, w, color="blue")
b6 = ax[0, 1].bar(x+w, df.Comp_events, w, color="black")

ax[0, 1].set_title("(b) Number of Heatwave Events")
ax[0, 1].set_ylabel("#Events")

ax[0, 1].set_xticks(x)
ax[0, 1].set_xticklabels(df.Division, rotation=30)

ax[0, 1].bar_label(b4, padding=3, fontsize=14)
ax[0, 1].bar_label(b5, padding=3, fontsize=14)
ax[0, 1].bar_label(b6, padding=3, fontsize=14)

# -----------------------------
# c Average duration
# -----------------------------

b7 = ax[1, 0].bar(x-w, df.Day_avg, w, color="red")
b8 = ax[1, 0].bar(x, df.Night_avg, w, color="blue")
b9 = ax[1, 0].bar(x+w, df.Comp_avg, w, color="black")

ax[1, 0].set_title("(c) Average Event Duration")
ax[1, 0].set_ylabel("#Days")

ax[1, 0].set_xticks(x)
ax[1, 0].set_xticklabels(df.Division, rotation=30)

ax[1, 0].bar_label(b7, fmt="%.1f", padding=3, fontsize=14)
ax[1, 0].bar_label(b8, fmt="%.1f", padding=3, fontsize=14)
ax[1, 0].bar_label(b9, fmt="%.1f", padding=3, fontsize=14)

# -----------------------------
# d Maximum duration
# -----------------------------

b10 = ax[1, 1].bar(x-w, df.Day_max, w, color="red")
b11 = ax[1, 1].bar(x, df.Night_max, w, color="blue")
b12 = ax[1, 1].bar(x+w, df.Comp_max, w, color="black")

ax[1, 1].set_title("(d) Maximum Event Duration")
ax[1, 1].set_ylabel("#Days")

ax[1, 1].set_xticks(x)
ax[1, 1].set_xticklabels(df.Division, rotation=30)

ax[1, 1].bar_label(b10, padding=3, fontsize=14)
ax[1, 1].bar_label(b11, padding=3, fontsize=14)
ax[1, 1].bar_label(b12, padding=3, fontsize=14)

# -----------------------------
# legend
# -----------------------------

fig.legend(
    loc='lower center',
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
    frameon=False
)

plt.tight_layout()
plt.show()






######-----Sample Fig 3------######
# ---------------------------------------------------
# 0 Import libraries
# ---------------------------------------------------
import xarray as xr
import numpy as np
import geopandas as gpd
import rioxarray
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib import ticker

# ---------------------------------------------------
# FIX MAP ASPECT FUNCTION
# ---------------------------------------------------
def fix_map_aspect(ax, gdf):
    xmin, ymin, xmax, ymax = gdf.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')

# ---------------------------------------------------
# 1 Load ERA5 files
# ---------------------------------------------------
files = [
    "D:/Masters Report/Report data_Final/2m temperature(1991-2010)_DayTemp.nc",
    "D:/Masters Report/Report data_Final/2m temperature(2011-2025)_DayTemp.nc",
    "D:/Masters Report/Report data_Final/2m temperature(1991-2010)_NightTemp.nc",
    "D:/Masters Report/Report data_Final/2m temperature(2011-2025)_NightTemp.nc"
]

datasets = [xr.open_dataset(f) for f in files]
ds = xr.concat(datasets, dim="valid_time").sortby("valid_time")

# ---------------------------------------------------
# 2 Convert temperature from Kelvin to Celsius
# ---------------------------------------------------
t2m = ds["t2m"] - 273.15

# ---------------------------------------------------
# 3 Fix ERA5 longitude
# ---------------------------------------------------
t2m = t2m.assign_coords(
    longitude=((t2m.longitude + 180) % 360) - 180
).sortby("longitude")

# ---------------------------------------------------
# 4 Set CRS
# ---------------------------------------------------
t2m = t2m.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
t2m = t2m.rio.write_crs("EPSG:4326")

# ---------------------------------------------------
# 5 Select JJA months (here: March, April, May)
# ---------------------------------------------------
jja = t2m.where(t2m.valid_time.dt.month.isin([3, 4, 5]), drop=True)

# ---------------------------------------------------
# 6 Separate day/night using MAX temperature window
# ---------------------------------------------------

# ---- Daytime (10:00–18:00) ----
day_hours = jja.where(
    (jja.valid_time.dt.hour >= 10) & (jja.valid_time.dt.hour <= 18),
    drop=True
)

day_daily = day_hours.resample(valid_time="1D").max()

# ---- Nighttime (23:00–06:00) ----
night_hours = jja.where(
    (jja.valid_time.dt.hour >= 23) | (jja.valid_time.dt.hour <= 6),
    drop=True
)

night_daily = night_hours.resample(valid_time="1D").min()

# ---- Mean & STD (for further analysis, not affecting HW detection)
day_mean = day_daily.mean(dim="valid_time")
day_std = day_daily.std(dim="valid_time")

night_mean = night_daily.mean(dim="valid_time")
night_std = night_daily.std(dim="valid_time")

# ---------------------------------------------------
# 7 Calculate heatwave thresholds (95th percentile)
# ---------------------------------------------------
day_thresh = day_daily.quantile(0.95, dim="valid_time")
night_thresh = night_daily.quantile(0.95, dim="valid_time")

# ---------------------------------------------------
# 8 Heatwave boolean arrays (WITH CONSECUTIVE CONDITION)
# ---------------------------------------------------

day_exceed = day_daily > day_thresh
night_exceed = night_daily > night_thresh

def consecutive_3days(mask):
    return mask.rolling(valid_time=3).sum() >= 3

day_hw = consecutive_3days(day_exceed)
night_hw = consecutive_3days(night_exceed)
compound_hw = day_hw & night_hw

# ---------------------------------------------------
# 9 Count heatwave days
# ---------------------------------------------------

day_hw_days = day_hw.sum(dim="valid_time")
night_hw_days = night_hw.sum(dim="valid_time")
comp_hw_days = compound_hw.sum(dim="valid_time")

# ---------------------------------------------------
# 10 Function to count heatwave events
# ---------------------------------------------------
def count_events(arr):
    events = np.zeros(arr.shape[1:])
    for i in range(arr.shape[1]):
        for j in range(arr.shape[2]):
            series = arr[:, i, j]
            length = 0
            count = 0
            for v in series:
                if v:
                    length += 1
                else:
                    if length >= 3:
                        count += 1
                    length = 0
            if length >= 3:
                count += 1
            events[i, j] = count
    return events

# ---------------------------------------------------
# 11 Calculate event maps
# ---------------------------------------------------
day_events = count_events(day_hw.values)
night_events = count_events(night_hw.values)
comp_events = count_events(compound_hw.values)

def to_xarray(events, template):
    return xr.DataArray(
        events,
        coords={"latitude": template.latitude,
                "longitude": template.longitude},
        dims=["latitude", "longitude"]
    )

day_events = to_xarray(day_events, day_hw)
night_events = to_xarray(night_events, night_hw)
comp_events = to_xarray(comp_events, compound_hw)

# ---------------------------------------------------
# 12 Load Bangladesh shapefile
# ---------------------------------------------------
bd = gpd.read_file(
    "D:/Masters Report/shape_file/gadm41_BGD_shp/gadm41_BGD_1.shp"
).to_crs("EPSG:4326")

# ---------------------------------------------------
# Set CRS for outputs
# ---------------------------------------------------
for data in [day_hw_days, night_hw_days, comp_hw_days,
             day_events, night_events, comp_events]:
    data.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
    data.rio.write_crs("EPSG:4326", inplace=True)

# ---------------------------------------------------
# Clip to Bangladesh
# ---------------------------------------------------
day_hw_days = day_hw_days.rio.clip(bd.geometry, bd.crs, drop=True)
night_hw_days = night_hw_days.rio.clip(bd.geometry, bd.crs, drop=True)
comp_hw_days = comp_hw_days.rio.clip(bd.geometry, bd.crs, drop=True)

day_events = day_events.rio.clip(bd.geometry, bd.crs, drop=True)
night_events = night_events.rio.clip(bd.geometry, bd.crs, drop=True)
comp_events = comp_events.rio.clip(bd.geometry, bd.crs, drop=True)

print("\n================ BANGLADESH SUMMARY STATISTICS ================\n")

def overall_stats(da, name):
    print(f"\n--- {name} ---")
    print("Mean :", float(da.mean().values))
    print("Std  :", float(da.std().values))
    print("Min  :", float(da.min().values))
    print("Max  :", float(da.max().values))
    print("Sum  :", float(da.sum().values))

overall_stats(day_hw_days, "Day Heatwave Days")
overall_stats(night_hw_days, "Night Heatwave Days")
overall_stats(comp_hw_days, "Compound Heatwave Days")

overall_stats(day_events, "Day Heatwave Events")
overall_stats(night_events, "Night Heatwave Events")
overall_stats(comp_events, "Compound Heatwave Events")

import pandas as pd

def zonal_stats(da, gdf, var_name):

    results = []

    for _, row in gdf.iterrows():
        geom = [row.geometry]

        clipped = da.rio.clip(geom, gdf.crs, drop=True)

        results.append({
            "Division": row["NAME_1"],
            "Mean": float(clipped.mean().values),
            "Max": float(clipped.max().values),
            "Min": float(clipped.min().values),
            "Std": float(clipped.std().values),
            "Sum": float(clipped.sum().values)
        })

    return pd.DataFrame(results)

day_hw_div = zonal_stats(day_hw_days, bd, "Day HW Days")
night_hw_div = zonal_stats(night_hw_days, bd, "Night HW Days")
comp_hw_div = zonal_stats(comp_hw_days, bd, "Compound HW Days")

day_event_div = zonal_stats(day_events, bd, "Day Events")
night_event_div = zonal_stats(night_events, bd, "Night Events")
comp_event_div = zonal_stats(comp_events, bd, "Compound Events")

with pd.ExcelWriter("D:/Masters Report/Report data_Final/Fig3_summary_statistics.xlsx") as writer:
    day_hw_div.to_excel(writer, sheet_name="Day_HW_Days", index=False)
    night_hw_div.to_excel(writer, sheet_name="Night_HW_Days", index=False)
    comp_hw_div.to_excel(writer, sheet_name="Comp_HW_Days", index=False)

    day_event_div.to_excel(writer, sheet_name="Day_Events", index=False)
    night_event_div.to_excel(writer, sheet_name="Night_Events", index=False)
    comp_event_div.to_excel(writer, sheet_name="Comp_Events", index=False)

# ---------------------------------------------------
# 13 Plot (FIXED MAP RATIO APPLIED)
# ---------------------------------------------------
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(8, 10))
gs = GridSpec(2, 3, width_ratios=[1, 1, 0.05], figure=fig)

# Row 1
ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())

# Get min/max dynamically
vmin_days = 0
vmax_days = max(float(day_hw_days.max()), float(night_hw_days.max()))

# Plot
p1 = day_hw_days.plot(ax=ax1, cmap="Reds", vmin=vmin_days, vmax=vmax_days, add_colorbar=False)
p2 = night_hw_days.plot(ax=ax2, cmap="Reds", vmin=vmin_days, vmax=vmax_days, add_colorbar=False)

bd.boundary.plot(ax=ax1, color="black", linewidth=1)
bd.boundary.plot(ax=ax2, color="black", linewidth=1)

fix_map_aspect(ax1, bd)
fix_map_aspect(ax2, bd)

ax1.set_title("Daytime Heatwave", fontsize=16)
ax2.set_title("Nighttime Heatwave", fontsize=16)

ax1.set_axis_off()
ax2.set_axis_off()

cax1 = fig.add_subplot(gs[0, 2])
cbar1 = fig.colorbar(p1, cax=cax1)
cbar1.set_label("Days", fontsize=16)
cbar1.ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# Row 2
ax3 = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
ax4 = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())

# ---- Get dynamic range ----
event_min = min(float(day_events.min()), float(night_events.min()))
event_max = max(float(day_events.max()), float(night_events.max()))

# ---- Plot ----
p3 = day_events.plot(ax=ax3, cmap="OrRd", vmin=event_min, vmax=event_max, add_colorbar=False)
p4 = night_events.plot(ax=ax4, cmap="OrRd", vmin=event_min, vmax=event_max, add_colorbar=False)
bd.boundary.plot(ax=ax3, color="black", linewidth=1)
bd.boundary.plot(ax=ax4, color="black", linewidth=1)

fix_map_aspect(ax3, bd)
fix_map_aspect(ax4, bd)

ax3.set_title("", fontsize=16)
ax4.set_title("", fontsize=16)

ax3.set_axis_off()
ax4.set_axis_off()

cax2 = fig.add_subplot(gs[1, 2])
cbar2 = fig.colorbar(p3, cax=cax2)
cbar2.set_label("Events", fontsize=16)
cbar2.ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

plt.subplots_adjust(wspace=0.002, hspace=0.2)
plt.show()



print("Max daytime HW days:", float(day_hw_days.max()))
print("Max nighttime HW days:", float(night_hw_days.max()))
print("Max compound HW days:", float(comp_hw_days.max()))
print("Max compound HW events:", float(comp_events.max()))




# ---------------------------------------------------
# Compound plots with space after first colorbar
# ---------------------------------------------------
fig = plt.figure(figsize=(10, 5))

# width_ratios: [first map, first cbar, spacer, second map, second cbar]
gs = GridSpec(1, 5, width_ratios=[1, 0.02, 0.05, 1, 0.02], figure=fig)  

# ---- Compound Heatwave Days ----
ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
p1 = comp_hw_days.plot(
    ax=ax1, cmap="Reds", vmin=0, vmax=float(comp_hw_days.max()), add_colorbar=False
)
bd.boundary.plot(ax=ax1, color="black", linewidth=1)
fix_map_aspect(ax1, bd)
ax1.set_title("Compound Heatwave Days", fontsize=16)
ax1.set_axis_off()

cax1 = fig.add_subplot(gs[0, 1])  # colorbar for first map
cbar1 = fig.colorbar(p1, cax=cax1)
cbar1.set_label("Days", fontsize=16)

# ---- Compound Heatwave Events ----
ax2 = fig.add_subplot(gs[0, 3], projection=ccrs.PlateCarree())  # second map after spacer
p2 = comp_events.plot(
    ax=ax2, cmap="OrRd", vmin=0, vmax=float(comp_events.max()), add_colorbar=False
)
bd.boundary.plot(ax=ax2, color="black", linewidth=1)
fix_map_aspect(ax2, bd)
ax2.set_title("Compound Heatwave Events", fontsize=16)
ax2.set_axis_off()

cax2 = fig.add_subplot(gs[0, 4])  # colorbar for second map
cbar2 = fig.colorbar(p2, cax=cax2)
cbar2.set_label("Events", fontsize=16)

plt.show()