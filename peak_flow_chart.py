import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import sys
import numpy as np
from scipy import stats
from scipy.interpolate import UnivariateSpline

# Argument parsing for CSV file path
parser = argparse.ArgumentParser(description="Plot peak flow readings.")
parser.add_argument('csv_file', nargs='?', default=r'E:/Downloads/peak flow2.csv', help='Path to CSV file containing peak flow readings (default: Downloads folder)')
args = parser.parse_args()
filepath = args.csv_file

# Load data (use default comma separator to correctly parse CSV)
try:
    data = pd.read_csv(filepath)
except FileNotFoundError:
    print(f"Error: File '{filepath}' not found.")
    sys.exit(1)

# Standardize column names: lowercase, spaces/hyphens to underscores
data.columns = data.columns.str.strip().str.lower() \
    .str.replace(' ', '_') \
    .str.replace('-', '_') \
    .str.replace(r"[()]", '', regex=True)

# Auto-detect date column if not explicitly named 'date'
if 'date' not in data.columns:
    date_candidates = [c for c in data.columns if 'date' in c]
    if date_candidates:
        data.rename(columns={date_candidates[0]: 'date'}, inplace=True)
    else:
        print(f"Error: No date column found. Available columns: {list(data.columns)}")
        sys.exit(1)

# Drop duplicates on available fields
subset_cols = [c for c in ['date', 'pef', 'fev_1', 'note'] if c in data.columns]
if subset_cols:
    data = data.drop_duplicates(subset=subset_cols)

# Convert date to datetime
data['date'] = pd.to_datetime(data['date'], format='%m/%d/%Y', errors='coerce')
if data['date'].isna().any():
    # Fallback to automatic parsing if format doesn't match
    data['date'] = pd.to_datetime(data['date'], errors='coerce')

# Remove rows with invalid dates or missing PEF values
data = data.dropna(subset=['date', 'pef'])

# Create numeric date for regression (days since start)
date_numeric = (data['date'] - data['date'].min()).dt.days

# Extract PEF numeric values (remove 'L/min')
data['pef'] = data['pef'].str.replace(' L/min', '').astype(float)

# Extract FEV-1 numeric values (remove 'L')
if 'fev_1' in data.columns:
    data['fev_1'] = data['fev_1'].str.replace(' L', '').astype(float)

# Split data into points with and without notes
no_notes = data[data['note'].isna()]
with_notes = data[data['note'].notna()]

# Create dual-axis plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot PEF on left axis
ax1.scatter(no_notes['date'], no_notes['pef'], c='blue', marker='o', label='PEF No Note')
ax1.scatter(with_notes['date'], with_notes['pef'], c='red', marker='*', s=100, label='PEF With Note')

# Add PEF regression line
pef_slope, pef_intercept, pef_r, pef_p, pef_se = stats.linregress(date_numeric, data['pef'])
pef_line = pef_intercept + pef_slope * date_numeric
ax1.plot(data['date'], pef_line, 'b--', alpha=0.8, linewidth=2, 
         label=f'PEF Linear (r²={pef_r**2:.3f}, p={pef_p:.3f})')

# Add PEF spline interpolation
valid_mask = ~(date_numeric.isna() | data['pef'].isna())
if valid_mask.sum() > 3:  # Need at least 4 points for spline
    sorted_indices = np.argsort(date_numeric[valid_mask])
    valid_date_numeric = date_numeric[valid_mask].iloc[sorted_indices]
    valid_pef = data['pef'][valid_mask].iloc[sorted_indices]
    
    pef_spline = UnivariateSpline(valid_date_numeric, valid_pef, s=len(valid_pef)*0.1)
    date_smooth = np.linspace(date_numeric.min(), date_numeric.max(), 100)
    pef_smooth = pef_spline(date_smooth)
    date_smooth_dt = data['date'].min() + pd.to_timedelta(date_smooth, unit='D')
    ax1.plot(date_smooth_dt, pef_smooth, 'b-', alpha=0.7, linewidth=3, label='PEF Spline')

ax1.set_xlabel('Date')
ax1.set_ylabel('PEF (L/min)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Plot FEV-1 on right axis if available
if 'fev_1' in data.columns:
    ax2 = ax1.twinx()
    ax2.scatter(no_notes['date'], no_notes['fev_1'], c='lightblue', marker='s', alpha=0.7, label='FEV-1 No Note')
    ax2.scatter(with_notes['date'], with_notes['fev_1'], c='orange', marker='D', s=100, alpha=0.7, label='FEV-1 With Note')
    
    # Add FEV-1 regression line
    fev1_slope, fev1_intercept, fev1_r, fev1_p, fev1_se = stats.linregress(date_numeric, data['fev_1'])
    fev1_line = fev1_intercept + fev1_slope * date_numeric
    ax2.plot(data['date'], fev1_line, 'orange', linestyle='--', alpha=0.8, linewidth=2,
             label=f'FEV-1 Linear (r²={fev1_r**2:.3f}, p={fev1_p:.3f})')
    
    # Add FEV-1 spline interpolation
    valid_fev1_mask = ~(date_numeric.isna() | data['fev_1'].isna())
    if valid_fev1_mask.sum() > 3:  # Need at least 4 points for spline
        fev1_sorted_indices = np.argsort(date_numeric[valid_fev1_mask])
        valid_fev1_date_numeric = date_numeric[valid_fev1_mask].iloc[fev1_sorted_indices]
        valid_fev1 = data['fev_1'][valid_fev1_mask].iloc[fev1_sorted_indices]
        
        fev1_spline = UnivariateSpline(valid_fev1_date_numeric, valid_fev1, s=len(valid_fev1)*0.1)
        fev1_smooth = fev1_spline(date_smooth)
        ax2.plot(date_smooth_dt, fev1_smooth, 'orange', alpha=0.7, linewidth=3, label='FEV-1 Spline')
    
    ax2.set_ylabel('FEV-1 (L)', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

# Add annotations for notes
for i, row in with_notes.iterrows():
    ax1.axvline(x=row['date'], color='gray', linestyle='--', alpha=0.5)
    ax1.text(row['date'], row['pef'] + 10, row['note'], rotation=90, verticalalignment='bottom')

# Customize
ax1.set_title('Peak Flow Readings with Linear & Spline Analysis: May-July 2025')
ax1.grid(True)
plt.xticks(rotation=45)

# Print regression statistics
print(f"\nLinear Regression Analysis:")
print(f"PEF: slope={pef_slope:.3f} L/min/day, r²={pef_r**2:.3f}, p-value={pef_p:.3f}")
if 'fev_1' in data.columns:
    print(f"FEV-1: slope={fev1_slope:.3f} L/day, r²={fev1_r**2:.3f}, p-value={fev1_p:.3f}")
print(f"\nSpline curves show smoothed trends that capture non-linear patterns in the data.")

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
if 'fev_1' in data.columns:
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
else:
    ax1.legend()

plt.tight_layout()

# Show plot
plt.show()