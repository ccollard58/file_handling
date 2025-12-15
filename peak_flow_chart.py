import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import sys
import numpy as np
from scipy import stats
from scipy.interpolate import UnivariateSpline
from treatment_analysis import analyze_treatments
from matplotlib.patches import Patch
import seaborn as sns

# Argument parsing for CSV file path
parser = argparse.ArgumentParser(description="Plot peak flow readings.")
parser.add_argument('csv_file', nargs='?', default=r'E:/Downloads/20251215.csv', help='Path to CSV file containing peak flow readings (default: Downloads folder)')
parser.add_argument('--annotate-treatments', action='store_true', help='Annotate chart with top-performing treatments based on adjusted means')
parser.add_argument('--annotate-model', choices=['ols', 'rf', 'both'], default='ols', help='Which model rankings to show in annotation')
parser.add_argument('--min-count', type=int, default=2, help='Minimum records per treatment for analysis (others grouped into "Other")')
parser.add_argument('--poly-day', type=int, default=0, help='Add polynomial day terms up to this degree for analysis')
parser.add_argument('--interact-day', action='store_true', help='Include treatment × time interactions in analysis')
parser.add_argument('--rf', action='store_true', help='Enable Random Forest analysis (requires scikit-learn)')
parser.add_argument('--rf-cv', action='store_true', help='Run time-series CV randomized search for RF and report best params')
parser.add_argument('--rf-cv-iter', type=int, default=10, help='Number of random parameter samples for RF CV')
parser.add_argument('--rf-cv-splits', type=int, default=3, help='Number of forward-chaining CV splits')
parser.add_argument(
    '--treatment-palette',
    default='husl',
    help=(
        'Palette/colormap name used to color treatment shading regions (default: husl). '
        'Accepts Seaborn palettes (e.g. husl, colorblind, deep, muted, pastel) and Matplotlib colormaps.'
    ),
)
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

# Extract PEF numeric values (remove 'L/min' if present)
if data['pef'].dtype == 'O':
    data['pef'] = data['pef'].str.replace(' L/min', '').astype(float)
else:
    data['pef'] = data['pef'].astype(float)

# Extract FEV-1 numeric values (remove 'L' if present)
if 'fev_1' in data.columns:
    if data['fev_1'].dtype == 'O':
        data['fev_1'] = data['fev_1'].str.replace(' L', '').astype(float)
    else:
        data['fev_1'] = data['fev_1'].astype(float)

# Prepare treatment info from notes (used for background shading)
has_note_col = 'note' in data.columns
if has_note_col:
    # Normalize treatment text; if two treatments are listed, keep them together as a combined label
    data['treatment'] = (
        data['note']
        .fillna('No treatment specified')
        .astype(str)
        .str.strip()
        .replace({'': 'No treatment specified'})
    )
else:
    data['treatment'] = 'No treatment column'

#############################
# Precompute smooth date grid
date_smooth = np.linspace(date_numeric.min(), date_numeric.max(), 100)
date_smooth_dt = data['date'].min() + pd.to_timedelta(date_smooth, unit='D')

# Compute regression stats upfront
pef_slope, pef_intercept, pef_r, pef_p, pef_se = stats.linregress(date_numeric, data['pef'])
if 'fev_1' in data.columns:
    fev1_slope, fev1_intercept, fev1_r, fev1_p, fev1_se = stats.linregress(date_numeric, data['fev_1'])

def _available_palette_names() -> list[str]:
    names: set[str] = set()
    try:
        from seaborn.palettes import SEABORN_PALETTES, MPL_PALETTES

        if isinstance(SEABORN_PALETTES, dict):
            names.update(SEABORN_PALETTES.keys())
        elif isinstance(SEABORN_PALETTES, (list, tuple, set)):
            names.update(SEABORN_PALETTES)

        if isinstance(MPL_PALETTES, dict):
            names.update(MPL_PALETTES.keys())
        elif isinstance(MPL_PALETTES, (list, tuple, set)):
            names.update(MPL_PALETTES)
    except Exception:
        pass

    try:
        names.update(plt.colormaps())
    except Exception:
        pass

    return sorted(n for n in names if isinstance(n, str) and n.strip())


def _print_available_palettes():
    palettes = _available_palette_names()
    if not palettes:
        print("Available palettes: (unable to enumerate on this environment)")
        return
    print("Available palettes/colormaps:")
    for name in palettes:
        print(f" - {name}")


def _make_palette(palette_name: str, n_colors: int):
    try:
        return sns.color_palette(palette_name, n_colors=n_colors)
    except Exception:
        try:
            cmap = plt.get_cmap(palette_name)
            if n_colors <= 1:
                return [cmap(0.5)]
            return [cmap(i / (n_colors - 1)) for i in range(n_colors)]
        except Exception:
            raise ValueError(f"Invalid palette: {palette_name}")


def compute_treatment_spans(df: pd.DataFrame, palette_name: str):
    if not has_note_col or df.empty:
        return [], {}, []
    df_shade = df.sort_values('date')[['date', 'treatment']].copy()
    segment_id = (df_shade['treatment'] != df_shade['treatment'].shift(1)).cumsum()
    df_shade['segment'] = segment_id
    treatments_in_order = df_shade.drop_duplicates('treatment')['treatment'].tolist()
    if len(treatments_in_order) > 0:
        try:
            palette = _make_palette(palette_name, len(treatments_in_order))
        except Exception:
            print(f"Warning: Invalid --treatment-palette '{palette_name}'.")
            _print_available_palettes()
            print("Falling back to default palette: husl")
            try:
                palette = _make_palette('husl', len(treatments_in_order))
            except Exception:
                base = plt.get_cmap('tab20').colors
                repeats = int(np.ceil(len(treatments_in_order) / len(base)))
                palette = (list(base) * repeats)[:len(treatments_in_order)]
    else:
        palette = []
    treatment_to_color = {treat: palette[i] for i, treat in enumerate(treatments_in_order)}
    min_date = df_shade['date'].min()
    max_date = df_shade['date'].max()
    seg_list = []
    for sid, g in df_shade.groupby('segment'):
        start = g['date'].min()
        end = g['date'].max()
        label = g['treatment'].iloc[0]
        seg_list.append((start, end, label))
    seg_list_sorted = sorted(seg_list, key=lambda x: x[0])
    spans = []
    for idx, (start, end, label) in enumerate(seg_list_sorted):
        span_end = seg_list_sorted[idx + 1][0] if idx < len(seg_list_sorted) - 1 else max_date
        spans.append((start, span_end, label))
    shading_handles = [Patch(facecolor=treatment_to_color[t], edgecolor='gray', linewidth=0.5, alpha=0.8, label=t) for t in treatments_in_order]
    return spans, treatment_to_color, shading_handles

def draw_spans(ax, spans, treatment_to_color):
    for start, end, label in spans:
        color = treatment_to_color.get(label, (0.85, 0.85, 0.85))
        ax.axvspan(start, end, facecolor=color, alpha=0.22, zorder=0)

spans, treatment_to_color, shading_handles = compute_treatment_spans(data, args.treatment_palette)

#############################
# Figure 1: PEF
fig1, ax_pef = plt.subplots(figsize=(12, 6))
draw_spans(ax_pef, spans, treatment_to_color)
ax_pef.scatter(data['date'], data['pef'], c='blue', marker='o', label='PEF')
pef_line = pef_intercept + pef_slope * date_numeric
ax_pef.plot(data['date'], pef_line, 'b--', alpha=0.8, linewidth=2, label=f'PEF Linear (r²={pef_r**2:.3f}, p={pef_p:.3f})')
valid_mask = ~(date_numeric.isna() | data['pef'].isna())
if valid_mask.sum() > 3:
    sorted_indices = np.argsort(date_numeric[valid_mask])
    valid_date_numeric = date_numeric[valid_mask].iloc[sorted_indices]
    valid_pef = data['pef'][valid_mask].iloc[sorted_indices]
    pef_spline = UnivariateSpline(valid_date_numeric, valid_pef, s=len(valid_pef)*0.1)
    pef_smooth = pef_spline(date_smooth)
    ax_pef.plot(date_smooth_dt, pef_smooth, 'b-', alpha=0.7, linewidth=3, label='PEF Spline')
ax_pef.set_xlabel('Date')
ax_pef.set_ylabel('PEF (L/min)')
start_dt = data['date'].min().date() if not data.empty else ''
end_dt = data['date'].max().date() if not data.empty else ''
ax_pef.set_title(f'PEF with Treatment Periods: {start_dt} — {end_dt}')
ax_pef.grid(True)
plt.setp(ax_pef.get_xticklabels(), rotation=45)
if shading_handles:
    ax_pef.legend([*ax_pef.get_legend_handles_labels()[0], *shading_handles],
                  [*ax_pef.get_legend_handles_labels()[1], *[h.get_label() for h in shading_handles]],
                  loc='lower right', framealpha=0.95)
else:
    ax_pef.legend(loc='lower right', framealpha=0.95)
plt.tight_layout()

#############################
# Figure 2: FEV-1
if 'fev_1' in data.columns:
    fig2, ax_fev = plt.subplots(figsize=(12, 6))
    draw_spans(ax_fev, spans, treatment_to_color)
    ax_fev.scatter(data['date'], data['fev_1'], c='orange', marker='D', s=60, alpha=0.7, label='FEV-1')
    fev1_line = fev1_intercept + fev1_slope * date_numeric
    ax_fev.plot(data['date'], fev1_line, 'orange', linestyle='--', alpha=0.8, linewidth=2, label=f'FEV-1 Linear (r²={fev1_r**2:.3f}, p={fev1_p:.3f})')
    valid_fev1_mask = ~(date_numeric.isna() | data['fev_1'].isna())
    if valid_fev1_mask.sum() > 3:
        fev1_sorted_indices = np.argsort(date_numeric[valid_fev1_mask])
        valid_fev1_date_numeric = date_numeric[valid_fev1_mask].iloc[fev1_sorted_indices]
        valid_fev1 = data['fev_1'][valid_fev1_mask].iloc[fev1_sorted_indices]
        fev1_spline = UnivariateSpline(valid_fev1_date_numeric, valid_fev1, s=len(valid_fev1)*0.1)
        fev1_smooth = fev1_spline(date_smooth)
        ax_fev.plot(date_smooth_dt, fev1_smooth, 'orange', alpha=0.7, linewidth=3, label='FEV-1 Spline')
    ax_fev.set_xlabel('Date')
    ax_fev.set_ylabel('FEV-1 (L)')
    ax_fev.set_title(f'FEV-1 with Treatment Periods: {start_dt} — {end_dt}')
    ax_fev.grid(True)
    plt.setp(ax_fev.get_xticklabels(), rotation=45)
    if shading_handles:
        ax_fev.legend([*ax_fev.get_legend_handles_labels()[0], *shading_handles],
                      [*ax_fev.get_legend_handles_labels()[1], *[h.get_label() for h in shading_handles]],
                      loc='lower right', framealpha=0.95)
    else:
        ax_fev.legend(loc='lower right', framealpha=0.95)
    plt.tight_layout()

print(f"\nLinear Regression Analysis:")
print(f"PEF: slope={pef_slope:.3f} L/min/day, r²={pef_r**2:.3f}, p-value={pef_p:.3f}")
if 'fev_1' in data.columns:
    print(f"FEV-1: slope={fev1_slope:.3f} L/day, r²={fev1_r**2:.3f}, p-value={fev1_p:.3f}")
print(f"\nSpline curves show smoothed trends that capture non-linear patterns in the data.")

# Annotate best-performing treatments (optional)
if args.annotate_treatments:
    try:
        model_choice = 'ols'
        if args.annotate_model == 'rf' and args.rf:
            model_choice = 'rf'
        elif args.annotate_model == 'both' and args.rf:
            model_choice = 'both'

        analysis = analyze_treatments(
            filepath,
            min_count=args.min_count,
            model='both' if args.rf else 'ols',
            poly_day=args.poly_day,
            interact_day=args.interact_day,
            rf_cv=args.rf_cv,
            rf_cv_iter=args.rf_cv_iter,
            rf_cv_splits=args.rf_cv_splits,
        )

        pef_adj = analysis.get('pef_adj') if model_choice in ('ols', 'both') else None
        pef_rf_adj = analysis.get('pef_rf_adj') if model_choice in ('rf', 'both') else None
        fev1_adj = analysis.get('fev1_adj') if model_choice in ('ols', 'both') else None
        fev1_rf_adj = analysis.get('fev1_rf_adj') if model_choice in ('rf', 'both') else None

        # Annotate PEF figure
        pef_lines = []
        if pef_adj is not None and not pef_adj.empty:
            top = pef_adj.head(3)
            pef_lines.append('Top PEF (OLS):')
            for _, r in top.iterrows():
                pef_lines.append(f" - {r['treatment']}: {r['adjusted_value']:.1f} L/min")
        if pef_rf_adj is not None and not pef_rf_adj.empty:
            top = pef_rf_adj.head(3)
            pef_lines.append('Top PEF (RF):')
            for _, r in top.iterrows():
                pef_lines.append(f" - {r['treatment']}: {r['adjusted_value']:.1f} L/min")
        if pef_lines:
            ax_pef.text(0.01, 0.02, '\n'.join(pef_lines), transform=ax_pef.transAxes,
                        fontsize=9, va='bottom', ha='left',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray'))

        # Annotate FEV-1 figure
        if 'fev_1' in data.columns:
            fev_lines = []
            if fev1_adj is not None and not fev1_adj.empty:
                top = fev1_adj.head(3)
                fev_lines.append('Top FEV-1 (OLS):')
                for _, r in top.iterrows():
                    fev_lines.append(f" - {r['treatment']}: {r['adjusted_value']:.3f} L")
            if fev1_rf_adj is not None and not fev1_rf_adj.empty:
                top = fev1_rf_adj.head(3)
                fev_lines.append('Top FEV-1 (RF):')
                for _, r in top.iterrows():
                    fev_lines.append(f" - {r['treatment']}: {r['adjusted_value']:.3f} L")
            if fev_lines:
                ax_fev.text(0.01, 0.02, '\n'.join(fev_lines), transform=ax_fev.transAxes,
                            fontsize=9, va='bottom', ha='left',
                            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray'))
    except Exception as e:
        print(f"Warning: treatment annotation failed: {e}")

# Show plot
plt.show()