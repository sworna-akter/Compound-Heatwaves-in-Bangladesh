# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 22:27:58 2026

@author: Admin
"""

####------Sample fig5-------######
import xarray as xr
import numpy as np
import geopandas as gpd
import rioxarray
import matplotlib.pyplot as plt

# ---------------------------------------------
# Function to fix map aspect
# ---------------------------------------------
def fix_map_aspect(ax, gdf):
    xmin, ymin, xmax, ymax = gdf.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')

# ---------------------------------------------
# 1. FILES
# ---------------------------------------------
files = [
    "D:/Masters Report/Report data_Final/2m temperature(1991-2010)_DayTemp.nc",
    "D:/Masters Report/Report data_Final/2m temperature(2011-2025)_DayTemp.nc",
    "D:/Masters Report/Report data_Final/2m temperature(1991-2010)_NightTemp.nc",
    "D:/Masters Report/Report data_Final/2m temperature(2011-2025)_NightTemp.nc"
]

shapefile = "D:/Masters Report/shape_file/gadm41_BGD_shp/gadm41_BGD_1.shp"

# ---------------------------------------------
# 2. LOAD DATA
# ---------------------------------------------
day_files = files[:2]
night_files = files[2:]

day_ds = xr.open_mfdataset(day_files, combine='by_coords', chunks={'valid_time': 300})
night_ds = xr.open_mfdataset(night_files, combine='by_coords', chunks={'valid_time': 300})

# Kelvin → Celsius
day_ds['t2m'] = day_ds['t2m'] - 273.15
night_ds['t2m'] = night_ds['t2m'] - 273.15

day_ds['t2m'] = day_ds['t2m'].where(day_ds['t2m'] > -100)
night_ds['t2m'] = night_ds['t2m'].where(night_ds['t2m'] > -100)

# ---------------------------------------------
# 3. SEPARATE DAY/NIGHT TEMPERATURE
# ---------------------------------------------
day_temp = day_ds['t2m'].where(
    (day_ds.valid_time.dt.hour >= 10) & (day_ds.valid_time.dt.hour <= 18),
    drop=True
).resample(valid_time='1D').max()

night_temp = night_ds['t2m'].where(
    (night_ds.valid_time.dt.hour >= 23) | (night_ds.valid_time.dt.hour <= 6),
    drop=True
).resample(valid_time='1D').min()

# 🔥 CRITICAL FIX: rechunk AFTER resampling
day_temp = day_temp.chunk({'valid_time': -1})
night_temp = night_temp.chunk({'valid_time': -1})

# ---------------------------------------------
# 4. THRESHOLDS (95th percentile)
# ---------------------------------------------
day_thr = day_temp.quantile(0.95, dim='valid_time')
night_thr = night_temp.quantile(0.95, dim='valid_time')

hw_day = day_temp > day_thr
hw_night = night_temp > night_thr

# ---------------------------------------------
# 5. 3-CONSECUTIVE DAYS FUNCTION
# ---------------------------------------------
def consecutive_3days(mask):
    return mask.rolling(valid_time=3, center=False).sum() >= 3

hw_day = consecutive_3days(hw_day)
hw_night = consecutive_3days(hw_night)
hw_compound = hw_day & hw_night

# ---------------------------------------------
# 6. STANDARDIZED ANOMALY COMPOSITE
# ---------------------------------------------
def composite(anom, mask):
    return anom.where(mask).mean(dim='valid_time')

day_mean = day_temp.mean(dim='valid_time')
day_std = day_temp.std(dim='valid_time')
night_mean = night_temp.mean(dim='valid_time')
night_std = night_temp.std(dim='valid_time')

day_anom = (day_temp - day_mean) / day_std
night_anom = (night_temp - night_mean) / night_std

a = composite(day_anom, hw_day)
b = composite(night_anom, hw_day)

c = composite(day_anom, hw_night)
d = composite(night_anom, hw_night)

e = composite(day_anom, hw_compound)
f = composite(night_anom, hw_compound)

# ---------------------------------------------
# 7. CLIP
# ---------------------------------------------
gdf = gpd.read_file(shapefile).to_crs("EPSG:4326")

def clip(data):
    data = data.rio.write_crs("EPSG:4326")
    return data.rio.clip(gdf.geometry, gdf.crs)

datasets = [clip(x) for x in [a, c, e, b, d, f]]

# ---------------------------------------------
# 9. BANGLADESH SUMMARY STATISTICS
# ---------------------------------------------
names = [
    "Day_HW_Day_Anom",
    "Night_HW_Day_Anom",
    "Compound_HW_Day_Anom",
    "Day_HW_Night_Anom",
    "Night_HW_Night_Anom",
    "Compound_HW_Night_Anom"
]

print("\n===== Bangladesh Summary Statistics =====\n")

for name, da in zip(names, datasets):
    vals = da.values.flatten()
    vals = vals[~np.isnan(vals)]

    print(f"{name}:")
    print(f"Mean = {np.mean(vals):.3f}")
    print(f"Max  = {np.max(vals):.3f}")
    print(f"Min  = {np.min(vals):.3f}")
    print(f"Std  = {np.std(vals):.3f}")
    print("----------------------------------")
    
    
    # ---------------------------------------------
# 10. DIVISION-WISE STATISTICS
# ---------------------------------------------
print("\n===== Division-wise Summary Statistics =====\n")

for name, da in zip(names, datasets):

    print(f"\n### {name} ###")

    for i, row in gdf.iterrows():
        geom = [row.geometry]
        div_name = row['NAME_1']

        clipped = da.rio.clip(geom, gdf.crs, drop=True)

        vals = clipped.values.flatten()
        vals = vals[~np.isnan(vals)]

        if len(vals) > 0:
            print(f"{div_name}: Mean={np.mean(vals):.3f}, Max={np.max(vals):.3f}, Min={np.min(vals):.3f}")
            
            
import pandas as pd

records = []

for name, da in zip(names, datasets):
    for i, row in gdf.iterrows():
        geom = [row.geometry]
        div_name = row['NAME_1']

        clipped = da.rio.clip(geom, gdf.crs, drop=True)
        vals = clipped.values.flatten()
        vals = vals[~np.isnan(vals)]

        if len(vals) > 0:
            records.append({
                "Variable": name,
                "Division": div_name,
                "Mean": np.mean(vals),
                "Max": np.max(vals),
                "Min": np.min(vals),
                "Std": np.std(vals)
            })

df = pd.DataFrame(records)
df.to_csv("D:/Masters Report/Report data_Final/Fig5_Division_Stats.csv", index=False)

# ---------------------------------------------
# 8. PLOT (3-COLUMN LAYOUT)
# ---------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(12, 10))
fig.subplots_adjust(
    left=0.05, right=0.98, top=0.92, bottom=0.01,
    wspace=0, hspace=0.01
)

plot_order = [datasets[0], datasets[1], datasets[2], datasets[3], datasets[4], datasets[5]]
titles = ["(a)", "(c)", "(e)", "(b)", "(d)", "(f)"]
vmin, vmax = 0, 2

for ax, data, title in zip(axes.flat, plot_order, titles):
    im = data.plot(
        ax=ax,
        cmap='RdYlBu_r',
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False
    )
    gdf.boundary.plot(ax=ax, color='black', linewidth=0.5)
    fix_map_aspect(ax, gdf)
    ax.set_axis_off()

axes[0, 0].set_title("Daytime HW Days", fontsize=16)
axes[0, 1].set_title("Nighttime HW Days", fontsize=16)
axes[0, 2].set_title("Compound HW Days", fontsize=16)
axes[1, 0].set_title("", fontsize=16)
axes[1, 1].set_title("", fontsize=16)
axes[1, 2].set_title("", fontsize=16)

fig.text(0.02, 0.75, "Daytime Composite", va='center', rotation='vertical', fontsize=16)
fig.text(0.02, 0.25, "Nighttime Composite", va='center', rotation='vertical', fontsize=16)

cbar_ax = fig.add_axes([0.25, -0.05, 0.5, 0.02])
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
cbar.set_label("T2m Standardized Anomaly Composite (°C)", fontsize=16)
cbar.ax.tick_params(labelsize=14)

plt.show()





