# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 22:24:13 2026

@author: Admin
"""

#####-------Sample Fig4----------########
import xarray as xr
import numpy as np
import geopandas as gpd
import rioxarray
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib import ticker
from scipy.stats import linregress
import pymannkendall as mk

# ------------------------------------------------
# Function to fix map aspect
# ------------------------------------------------
def fix_map_aspect(ax, gdf):
    xmin, ymin, xmax, ymax = gdf.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')

# ------------------------------------------------
# 1. Load ERA5 NetCDF files
# ------------------------------------------------
files = [
    "D:/Masters Report/Report data_Final/2m temperature(1991-2010)_DayTemp.nc",
    "D:/Masters Report/Report data_Final/2m temperature(2011-2025)_DayTemp.nc",
    "D:/Masters Report/Report data_Final/2m temperature(1991-2010)_NightTemp.nc",
    "D:/Masters Report/Report data_Final/2m temperature(2011-2025)_NightTemp.nc"
]

ds = xr.open_mfdataset(files, combine="nested", concat_dim="valid_time")
ds = ds.sortby("valid_time")                 # Ensure time is strictly increasing


# Kelvin → Celsius
ds["t2m"] = ds["t2m"] - 273.15

# ------------------------------------------------
# 2. Select season
# ------------------------------------------------
ds = ds.sel(valid_time=ds.valid_time.dt.month.isin([3, 4, 5]))

# ------------------------------------------------
# 3. Day / Night temperature
# ------------------------------------------------

# Daytime: 10:00–18:00
day_hours = ds.t2m.where(
    (ds.valid_time.dt.hour >= 10) & (ds.valid_time.dt.hour <= 18),
    drop=True
)
tmax_day = day_hours.resample(valid_time="1D").max()

# Nighttime: 23:00–06:00 (cross midnight)
night_hours = ds.t2m.where(
    (ds.valid_time.dt.hour >= 23) | (ds.valid_time.dt.hour <= 6),
    drop=True
)
tmax_night = night_hours.resample(valid_time="1D").min()

tmax_day = tmax_day.chunk({"valid_time": -1})
tmax_night = tmax_night.chunk({"valid_time": -1})

# ------------------------------------------------
# 4. Thresholds
# ------------------------------------------------
tmax_day_thr = tmax_day.quantile(0.95, dim="valid_time")
tmax_night_thr = tmax_night.quantile(0.95, dim="valid_time")

# ------------------------------------------------
# 5. Exceedance masks
# ------------------------------------------------
day_exceed = tmax_day > tmax_day_thr
night_exceed = tmax_night > tmax_night_thr
compound_exceed = day_exceed & night_exceed

# ------------------------------------------------
# 6. Heatwave detection (≥3 consecutive days)
# ------------------------------------------------
def detect_hw(mask):
    # Rolling sum over 3 days
    rolling = mask.rolling(valid_time=3, center=False).sum()
    # Only mark days where rolling sum >=3
    hw = rolling >= 3
    # Include all days in the sequence
    hw_full = hw | hw.shift(valid_time=1).fillna(False) | hw.shift(valid_time=2).fillna(False)
    return hw_full

day_hw = detect_hw(day_exceed)
night_hw = detect_hw(night_exceed)
compound_hw = detect_hw(compound_exceed)
# ------------------------------------------------
# 7. Yearly heatwave days
# ------------------------------------------------
day_year = day_hw.groupby("valid_time.year").sum().compute()
night_year = night_hw.groupby("valid_time.year").sum().compute()
compound_year = compound_hw.groupby("valid_time.year").sum().compute()

years = day_year.year.values

# ------------------------------------------------
# 8. Trend calculation
# ------------------------------------------------
def calc_trend(data):

    y = np.array(data.values, dtype=float)
    ny, nlat, nlon = y.shape

    trend = np.full((nlat, nlon), np.nan)

    for i in range(nlat):
        for j in range(nlon):

            series = y[:, i, j]

            if np.all(np.isnan(series)):
                continue

            valid = ~np.isnan(series)

            if valid.sum() < 3:
                continue

            slope, _, _, _, _ = linregress(years[valid], series[valid])
            trend[i, j] = slope * 10

    return trend

trend_day = calc_trend(day_year)
trend_night = calc_trend(night_year)
trend_compound = calc_trend(compound_year)

# ------------------------------------------------
# 9. Mann-Kendall significance (kept but not plotted)
# ------------------------------------------------
def mk_test(data):

    y = np.array(data.values, dtype=float)
    ny, nlat, nlon = y.shape

    sig = np.zeros((nlat, nlon))

    for i in range(nlat):
        for j in range(nlon):

            series = y[:, i, j]

            if np.all(np.isnan(series)):
                continue

            valid = ~np.isnan(series)

            if valid.sum() < 3:
                continue

            result = mk.original_test(series[valid])

            if result.p < 0.05:
                sig[i, j] = 1

    return sig

sig_day = mk_test(day_year)
sig_night = mk_test(night_year)
sig_compound = mk_test(compound_year)

# ------------------------------------------------
# 10. Bangladesh shapefile
# ------------------------------------------------
bd = gpd.read_file(
    "D:/Masters Report/shape_file/gadm41_BGD_shp/gadm41_BGD_1.shp"
).to_crs("EPSG:4326")

# ------------------------------------------------
# 11. Convert trends to xarray for clipping
# ------------------------------------------------
lat = ds.latitude
lon = ds.longitude

def to_xarray_and_clip(data, lat, lon, shape):

    da = xr.DataArray(
        data,
        coords=[lat, lon],
        dims=["latitude", "longitude"]
    )

    da = da.rio.write_crs("EPSG:4326")

    return da.rio.clip(shape.geometry, shape.crs)

trend_day = to_xarray_and_clip(trend_day, lat, lon, bd)
trend_night = to_xarray_and_clip(trend_night, lat, lon, bd)
trend_compound = to_xarray_and_clip(trend_compound, lat, lon, bd)



# ---------------------------------------------
# 11B. BANGLADESH-WIDE SUMMARY STATISTICS
# ---------------------------------------------

def summary_stats(da, name):
    data = da.values
    data = data[~np.isnan(data)]

    print(f"\n===== {name} (Bangladesh-wide) =====")
    print(f"Mean trend (days/decade): {np.mean(data):.3f}")
    print(f"Median trend: {np.median(data):.3f}")
    print(f"Std deviation: {np.std(data):.3f}")
    print(f"Min: {np.min(data):.3f}")
    print(f"Max: {np.max(data):.3f}")
    print(f"5th percentile: {np.percentile(data, 5):.3f}")
    print(f"95th percentile: {np.percentile(data, 95):.3f}")
    print(f"Positive grid cells (%): {np.sum(data > 0) / len(data) * 100:.2f}")
    print(f"Negative grid cells (%): {np.sum(data < 0) / len(data) * 100:.2f}")


summary_stats(trend_day, "Daytime Heatwave Trend")
summary_stats(trend_night, "Nighttime Heatwave Trend")
summary_stats(trend_compound, "Compound Heatwave Trend")

# ---------------------------------------------
# 11C. DIVISION-WISE SUMMARY STATISTICS
# ---------------------------------------------

def division_stats(da, gdf, name):
    print(f"\n===== {name} (Division-wise) =====")

    for idx, row in gdf.iterrows():

        geom = [row.geometry]
        clipped = da.rio.clip(geom, gdf.crs)

        values = clipped.values
        values = values[~np.isnan(values)]

        if len(values) == 0:
            continue

        print(f"\n{row['NAME_1']}:")
        print(f"Mean: {np.mean(values):.3f}")
        print(f"Median: {np.median(values):.3f}")
        print(f"Min: {np.min(values):.3f}")
        print(f"Max: {np.max(values):.3f}")
        print(f"Std: {np.std(values):.3f}")
        print(f"Positive area (%): {np.sum(values > 0)/len(values)*100:.2f}")


division_stats(trend_day, bd, "Daytime Heatwave Trend")
division_stats(trend_night, bd, "Nighttime Heatwave Trend")
division_stats(trend_compound, bd, "Compound Heatwave Trend")
# ------------------------------------------------
# 12. Plot maps (FIXED RATIO + NO WHITE DOTS)
# ------------------------------------------------
fig, axes = plt.subplots(
    1, 3,
    figsize=(9, 5),
    subplot_kw={"projection": ccrs.PlateCarree()}
)

titles = ["Daytime Heatwaves", "Nighttime Heatwaves", "Compound Heatwaves"]
datasets = [trend_day, trend_night, trend_compound]

for i in range(3):

    ax = axes[i]

    im = ax.pcolormesh(
        datasets[i].longitude,
        datasets[i].latitude,
        datasets[i],
        cmap="RdBu_r",
        vmin=-1,
        vmax=3,
        shading="auto",
        transform=ccrs.PlateCarree()
    )

    bd.boundary.plot(ax=ax, color="black", linewidth=0.7)

    # ✅ FIXED MAP RATIO
    fix_map_aspect(ax, bd)

    ax.set_title(titles[i], fontsize=15)

# ------------------------------------------------
# 13. Colorbar
# ------------------------------------------------
cbar_ax = fig.add_axes([0.19, 0.05, 0.65, 0.03])

cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")

cbar.set_label("(Days/Decade)", fontsize=14)

plt.subplots_adjust(bottom=0, wspace=0.1)

plt.show()
