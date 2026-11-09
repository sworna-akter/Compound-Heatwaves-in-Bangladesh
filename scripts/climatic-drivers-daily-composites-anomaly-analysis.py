# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 22:28:26 2026

@author: Admin
"""

#######-----Sample Fig 8------#####
import xarray as xr
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import rioxarray

# ---------------------------------------------
# FUNCTION TO FIX MAP ASPECT
# ---------------------------------------------
def fix_map_aspect(ax, gdf):
    xmin, ymin, xmax, ymax = gdf.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')

# ---------------------------------------------
# 1. FILE PATHS
# ---------------------------------------------
temp_files = [
    "D:/Masters Report/Report data_Final/2m temperature(1991-2010)_DayTemp.nc",
    "D:/Masters Report/Report data_Final/2m temperature(2011-2025)_DayTemp.nc",
    "D:/Masters Report/Report data_Final/2m temperature(1991-2010)_NightTemp.nc",
    "D:/Masters Report/Report data_Final/2m temperature(2011-2025)_NightTemp.nc"
]

cloud_file = "D:/Masters Report/Report data_Final/Total-cloud-cover(1991-2025).nc"
sw_file = "D:/Masters Report/Report data_Final/Surface solar radiation downward, clear sky&all sky.nc"
lw_file = "D:/Masters Report/Report data_Final/Surface thermal radiation downward, clear sky&all sky.nc"

shp_path = "D:/Masters Report/shape_file/gadm41_BGD_shp/gadm41_BGD_1.shp"

# ---------------------------------------------
# 2. LOAD DATA
# ---------------------------------------------
day_ds = xr.open_mfdataset(temp_files[:2], combine='by_coords')
night_ds = xr.open_mfdataset(temp_files[2:], combine='by_coords')

ds_cloud = xr.open_dataset(cloud_file)
ds_sw = xr.open_dataset(sw_file)
ds_lw = xr.open_dataset(lw_file)

# ---------------------------------------------
# 3. FIX COORDS + TIME
# ---------------------------------------------
def fix(ds):
    if 'valid_time' in ds.dims:
        ds = ds.rename({'valid_time': 'time'})
    if 'longitude' in ds.coords:
        ds = ds.rename({'longitude': 'lon'})
    if 'latitude' in ds.coords:
        ds = ds.rename({'latitude': 'lat'})
    return ds

day_ds = fix(day_ds)
night_ds = fix(night_ds)
ds_cloud = fix(ds_cloud)
ds_sw = fix(ds_sw)
ds_lw = fix(ds_lw)

# ---------------------------------------------
# 4. VARIABLES
# ---------------------------------------------
day_ds['t2m'] = day_ds['t2m'] - 273.15
night_ds['t2m'] = night_ds['t2m'] - 273.15

tcc = ds_cloud['tcc']
sw_cre = ds_sw['ssrd'] - ds_sw['ssrdc']
lw_cre = ds_lw['strd'] - ds_lw['strdc']

# ---------------------------------------------
# 5. CLIP
# ---------------------------------------------
gdf = gpd.read_file(shp_path).to_crs("EPSG:4326")

def clip(da):
    return da.rio.write_crs("EPSG:4326").rio.clip(gdf.geometry, gdf.crs)

day_t2m = clip(day_ds['t2m']).compute()
night_t2m = clip(night_ds['t2m']).compute()
tcc = clip(tcc).compute()
sw_cre = clip(sw_cre).compute()
lw_cre = clip(lw_cre).compute()

# ---------------------------------------------
# 6. DAY/NIGHT TEMPERATURE (FIG 5 LOGIC)
# ---------------------------------------------
day_temp = day_t2m.where(
    (day_t2m.time.dt.hour >= 10) & (day_t2m.time.dt.hour <= 18),
    drop=True
).resample(time='1D').max()

night_temp = night_t2m.where(
    (night_t2m.time.dt.hour >= 23) | (night_t2m.time.dt.hour <= 6),
    drop=True
).resample(time='1D').min()

# ---------------------------------------------
# 7. HEATWAVE MASKS
# ---------------------------------------------
tmax_thr = day_temp.quantile(0.95, dim='time')
tmin_thr = night_temp.quantile(0.95, dim='time')

def consecutive(mask):
    return mask.rolling(time=3).sum() >= 3

hw_day = consecutive(day_temp > tmax_thr)
hw_night = consecutive(night_temp > tmin_thr)
hw_compound = hw_day & hw_night

# ---------------------------------------------
# 8. DAILY VARIABLES
# ---------------------------------------------
tcc_d = tcc.resample(time='1D').mean()
sw_d = sw_cre.resample(time='1D').mean()
lw_d = lw_cre.resample(time='1D').mean()

# ---------------------------------------------
# 9. ANOMALIES (PHYSICAL CRE INTERPRETATION)
# ---------------------------------------------
tcc_anom = tcc_d - tcc_d.mean('time')
sw_anom = sw_d - sw_d.mean('time')
lw_anom = lw_d - lw_d.mean('time')

# ---------------------------------------------
# 10. COMPOSITE FUNCTION
# ---------------------------------------------
def comp(data, mask):
    return data.where(mask).mean('time')

# ---------------------------------------------
# 11. COMPOSITES
# ---------------------------------------------
tcc_a = comp(tcc_anom, hw_day)
tcc_b = comp(tcc_anom, hw_night)
tcc_c = comp(tcc_anom, hw_compound)

sw_a = comp(sw_anom, hw_day)
sw_b = comp(sw_anom, hw_night)
sw_c = comp(sw_anom, hw_compound)

lw_a = comp(lw_anom, hw_day)
lw_b = comp(lw_anom, hw_night)
lw_c = comp(lw_anom, hw_compound)

# ---------------------------------------------
# 12. PLOT
# ---------------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(9, 14))

lev_cloud = np.linspace(-0.2, 0.2, 21)
lev_sw = np.linspace(np.nanmin(sw_anom), np.nanmax(sw_anom), 21)
lev_lw = np.linspace(np.nanmin(lw_anom), np.nanmax(lw_anom), 21)

# ROW 1: CLOUD
im1 = axes[0,0].contourf(tcc_a.lon, tcc_a.lat, tcc_a, levels=lev_cloud, cmap='RdBu_r')
im2 = axes[0,1].contourf(tcc_b.lon, tcc_b.lat, tcc_b, levels=lev_cloud, cmap='RdBu_r')
im3 = axes[0,2].contourf(tcc_c.lon, tcc_c.lat, tcc_c, levels=lev_cloud, cmap='RdBu_r')

# ROW 2: SW CRE (COLUMN 2 REMOVED)
im4 = axes[1,0].contourf(sw_a.lon, sw_a.lat, sw_a, levels=lev_sw, cmap='RdBu_r')

axes[1,1].cla()
axes[1,1].set_axis_off()

im5 = axes[1,2].contourf(sw_c.lon, sw_c.lat, sw_c, levels=lev_sw, cmap='RdBu_r')

# ROW 3: LW CRE
im6 = axes[2,0].contourf(lw_a.lon, lw_a.lat, lw_a, levels=lev_lw, cmap='RdBu_r')
im7 = axes[2,1].contourf(lw_b.lon, lw_b.lat, lw_b, levels=lev_lw, cmap='RdBu_r')
im8 = axes[2,2].contourf(lw_c.lon, lw_c.lat, lw_c, levels=lev_lw, cmap='RdBu_r')

# ---------------------------------------------
# 13. BORDERS + ASPECT
# ---------------------------------------------
for ax in axes.flat:
    if ax is not axes[1,1]:   # 🔴 skip blank panel completely
        gdf.boundary.plot(ax=ax, color='black', linewidth=0.5)
        fix_map_aspect(ax, gdf)
        ax.set_xticks([])
        ax.set_yticks([])
# ---------------------------------------------
# 14. TITLES
# ---------------------------------------------
axes[0,0].set_title("Day HW")
axes[0,1].set_title("Night HW")
axes[0,2].set_title("Compound HW")

axes[0,0].set_ylabel("Cloud Fraction")
axes[1,0].set_ylabel("Shortwave CRE")
axes[2,0].set_ylabel("Longwave CRE")

# ---------------------------------------------
# 15. ROW COLORBARS (FIXED PER ROW)
# ---------------------------------------------
plt.subplots_adjust(
    left=0.05,
    right=0.98,
    top=0.95,
    bottom=0.08,
    hspace=0.4,
    wspace=0
)

# Get positions of each row
row1_pos = axes[0,0].get_position()
row2_pos = axes[1,0].get_position()
row3_pos = axes[2,0].get_position()

# --- Row 1 colorbar (Cloud) ---
cbar1_ax = fig.add_axes([
    0.15, 
    row1_pos.y0 - 0.030,   # just below row 1
    0.7, 
    0.010
])
cbar1 = fig.colorbar(im3, cax=cbar1_ax, orientation='horizontal')
cbar1.set_label("Cloud Fraction Anomaly")

# --- Row 2 colorbar (SW CRE) ---
cbar2_ax = fig.add_axes([
    0.15, 
    row2_pos.y0 - 0.030,   # just below row 2
    0.7, 
    0.010
])
cbar2 = fig.colorbar(im4, cax=cbar2_ax, orientation='horizontal')
cbar2.set_label("Shortwave CRE Anomaly (W/m²)")

# --- Row 3 colorbar (LW CRE) ---
cbar3_ax = fig.add_axes([
    0.15, 
    row3_pos.y0 - 0.030,   # just below row 3
    0.7, 
    0.010
])
cbar3 = fig.colorbar(im6, cax=cbar3_ax, orientation='horizontal')
cbar3.set_label("Longwave CRE Anomaly (W/m²)")

plt.show()
