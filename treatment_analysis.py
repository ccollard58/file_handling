import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# try:
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
SKLEARN_AVAILABLE = True
# except Exception:
    # SKLEARN_AVAILABLE = False


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_').str.replace(r"[()]", '', regex=True)
    )
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    if 'date' not in df.columns:
        date_candidates = [c for c in df.columns if 'date' in c]
        if date_candidates:
            df = df.rename(columns={date_candidates[0]: 'date'})
        else:
            raise ValueError(f"No date column found. Available columns: {list(df.columns)}")

    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')
    if df['date'].isna().any():
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    df = df.dropna(subset=['date'])
    return df


def to_numeric(series: pd.Series, remove_suffix: str | None = None) -> pd.Series:
    s = series.copy()
    if s.dtype == 'O':
        if remove_suffix:
            s = s.str.replace(remove_suffix, '')
        s = pd.to_numeric(s, errors='coerce')
    else:
        s = pd.to_numeric(s, errors='coerce')
    return s


def normalize_treatment_from_note(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'note' in df.columns:
        tr = df['note'].fillna('No treatment specified').astype(str).str.strip()
        tr = tr.replace({'': 'No treatment specified'})
    else:
        tr = pd.Series(['No treatment column'] * len(df), index=df.index)

    # Optional: normalize separators for combined therapies
    # Replace common connectors with "+"; keep original case for readability
    tr_norm = (
        tr.str.replace(r"\s*/\s*", ' + ', regex=True)
          .str.replace(r"\s*\+\s*", ' + ', regex=True)
          .str.replace(r"\s*&\s*", ' + ', regex=True)
          .str.replace(r"\s*and\s*", ' + ', regex=True)
          .str.replace(r"\s*,\s*", ' + ', regex=True)
          .str.replace(r"\s*;\s*", ' + ', regex=True)
          .str.replace(r"\s+", ' ', regex=True)
          .str.strip()
    )

    df['treatment'] = tr_norm
    return df


def build_design(
    df: pd.DataFrame,
    value_col: str,
    min_count_per_treat: int = 2,
    poly_day: int = 0,
    interact_day: bool = False,
):
    # Keep only rows with the target value present
    d = df[['date', 'treatment', value_col]].dropna(subset=[value_col]).copy()
    if d.empty:
        return None, None, None, None

    # Time regressors
    d['day'] = (d['date'] - d['date'].min()).dt.days.astype(float)
    # Polynomial day terms
    poly_cols = []
    for k in range(2, poly_day + 1):
        col = f'day{k}'
        d[col] = d['day'] ** k
        poly_cols.append(col)

    # Choose baseline as most frequent treatment; rare treatments grouped to 'Other'
    counts = d['treatment'].value_counts()
    common = counts[counts >= min_count_per_treat].index.tolist()
    d.loc[~d['treatment'].isin(common), 'treatment'] = 'Other'

    # Recompute counts after grouping
    counts = d['treatment'].value_counts()
    baseline = counts.idxmax()

    # Create dummies excluding baseline
    dummies = pd.get_dummies(d['treatment'], prefix='tr', drop_first=False)
    if f'tr_{baseline}' in dummies.columns:
        dummies = dummies.drop(columns=[f'tr_{baseline}'])
    # If all rows are baseline (no other dummies), just use intercept + day (+ polynomials)
    X_list = [np.ones((len(d), 1)), d['day'].to_numpy().reshape(-1, 1)]
    for col in poly_cols:
        X_list.append(d[col].to_numpy().reshape(-1, 1))
    if dummies.shape[1] > 0:
        X_list.append(dummies.to_numpy(dtype=float))
    # Optional interactions: treatment × day and × polynomial day terms
    inter_cols = []
    if interact_day and dummies.shape[1] > 0:
        for tr_col in dummies.columns:
            # tr_* : day
            X_list.append((d['day'].to_numpy() * dummies[tr_col].to_numpy()).reshape(-1, 1))
            inter_cols.append(f'{tr_col}:day')
            # tr_* : day^k
            for col in poly_cols:
                X_list.append((d[col].to_numpy() * dummies[tr_col].to_numpy()).reshape(-1, 1))
                inter_cols.append(f'{tr_col}:{col}')
    X = np.hstack(X_list)
    y = d[value_col].to_numpy(dtype=float)

    # Column names
    columns = ['Intercept', 'day'] + poly_cols + [c for c in dummies.columns] + inter_cols
    return X, y, columns, baseline, d


def ols_fit(X: np.ndarray, y: np.ndarray):
    n, p = X.shape
    # Solve OLS via least squares
    beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    # Compute residuals explicitly if numpy didn't return them (when n <= p)
    y_hat = X @ beta
    e = y - y_hat
    sse = float(e.T @ e)
    dof = max(n - p, 1)
    mse = sse / dof
    # Variance-covariance
    XtX_inv = np.linalg.pinv(X.T @ X)
    var_beta = mse * XtX_inv
    se_beta = np.sqrt(np.diag(var_beta))
    with np.errstate(divide='ignore', invalid='ignore'):
        t_stats = beta / se_beta
    # Two-sided p-values using t-distribution
    p_vals = 2 * stats.t.sf(np.abs(t_stats), df=dof)
    # R^2
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - sse / sst if sst > 0 else np.nan
    return {
        'beta': beta,
        'se': se_beta,
        't': t_stats,
        'p': p_vals,
        'r2': r2,
        'mse': mse,
        'dof': dof,
        'y_hat': y_hat,
    }


def adjusted_means(result, columns, baseline_label: str, day_ref: float):
    """Compute adjusted means with 95% CIs and p-values for each treatment."""
    beta = result['beta']
    se_beta = result['se']
    p_vals = result['p']
    dof = result['dof']
    mse = result['mse']
    col_index = {c: i for i, c in enumerate(columns)}
    tr_cols = [c for c in columns if c.startswith('tr_')]

    # Variance-covariance matrix for computing SE of predicted values
    # Approximation: use result['mse'] * (X'X)^-1 from fitting
    # For simplicity, compute treatment-specific CIs using SE of the treatment coefficient
    # The CI for a non-baseline treatment's adjusted mean can be approximated as:
    #   baseline_mean ± CI  +  treatment_effect ± CI_effect
    # but precisely requires the full covariance. We'll use coefficient SE directly.

    def predict_for_treat(t_label: str):
        x = np.zeros(len(columns), dtype=float)
        if 'Intercept' in col_index:
            x[col_index['Intercept']] = 1.0
        if 'day' in col_index:
            x[col_index['day']] = day_ref
        for k in range(2, 10):
            name = f'day{k}'
            if name in col_index:
                x[col_index[name]] = day_ref ** k
        if t_label != baseline_label:
            tr_name = f'tr_{t_label}'
            if tr_name in col_index:
                x[col_index[tr_name]] = 1.0
        if t_label != baseline_label:
            name = f'tr_{t_label}:day'
            if name in col_index:
                x[col_index[name]] = day_ref
            for k in range(2, 10):
                name = f'tr_{t_label}:day{k}'
                if name in col_index:
                    x[col_index[name]] = day_ref ** k
        return float(np.dot(x, beta)), x

    t_crit = stats.t.ppf(0.975, df=dof) if dof > 0 else 1.96
    treatments = [baseline_label] + [c.replace('tr_', '') for c in tr_cols]
    rows = []
    for t in treatments:
        adj_mean, x_vec = predict_for_treat(t)
        # For treatment effect coefficient, get its SE and p-value
        if t == baseline_label:
            # Baseline: effect is zero by definition; use intercept p-value for reference
            effect = 0.0
            se_effect = 0.0
            p_val = np.nan
            ci_lower = adj_mean
            ci_upper = adj_mean
        else:
            tr_name = f'tr_{t}'
            if tr_name in col_index:
                idx = col_index[tr_name]
                effect = float(beta[idx])
                se_effect = float(se_beta[idx])
                p_val = float(p_vals[idx])
                ci_lower = adj_mean - t_crit * se_effect
                ci_upper = adj_mean + t_crit * se_effect
            else:
                effect = 0.0
                se_effect = 0.0
                p_val = np.nan
                ci_lower = adj_mean
                ci_upper = adj_mean
        rows.append({
            'treatment': t,
            'reference_day': day_ref,
            'adjusted_mean': adj_mean,
            'effect': effect,
            'se': se_effect,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_val,
            'significant': p_val < 0.05 if not np.isnan(p_val) else False,
        })
    return pd.DataFrame(rows).sort_values('adjusted_mean', ascending=False)


def rf_fit_with_time_split(X: np.ndarray, y: np.ndarray, columns: list[str], dates: pd.Series,
                           test_fraction: float = 0.3, random_state: int = 42,
                           n_estimators: int = 400, max_depth: int | None = None):
    # Remove intercept column if present
    col_index = {c: i for i, c in enumerate(columns)}
    keep_idx = [i for i, c in enumerate(columns) if c != 'Intercept']
    X_use = X[:, keep_idx]
    feat_names = [c for c in columns if c != 'Intercept']

    # Time-aware split: sort by date
    order = np.argsort(dates.to_numpy().astype('datetime64[ns]'))
    X_sorted = X_use[order]
    y_sorted = y[order]
    n = len(y_sorted)
    n_test = max(int(np.floor(test_fraction * n)), 1)
    n_train = max(n - n_test, 1)
    X_train, y_train = X_sorted[:n_train], y_sorted[:n_train]
    X_test, y_test = X_sorted[n_train:], y_sorted[n_train:]

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        oob_score=False,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test) if len(X_test) > 0 else np.array([])
    r2 = r2_score(y_test, y_pred) if len(y_pred) > 0 else np.nan
    mae = mean_absolute_error(y_test, y_pred) if len(y_pred) > 0 else np.nan
    return {
        'model': model,
        'feature_names': feat_names,
        'r2_test': float(r2) if r2 == r2 else None,  # handle nan
        'mae_test': float(mae) if mae == mae else None,
        'n_train': int(n_train),
        'n_test': int(len(y_test)),
        'keep_idx': keep_idx,
    }


def rf_adjusted_means(rf_result: dict, columns: list[str], baseline_label: str, all_treatments: list[str], day_ref: float):
    """Compute RF adjusted means. p-values not directly available for RF."""
    feat_names = rf_result['feature_names']
    base = {name: 0.0 for name in feat_names}
    for name in feat_names:
        if name == 'day':
            base[name] = day_ref
        elif name.startswith('day') and name[3:].isdigit():
            k = int(name[3:])
            base[name] = day_ref ** k

    rows = []
    for t in [baseline_label] + [t for t in all_treatments if t != baseline_label]:
        x = base.copy()
        if t != baseline_label:
            key = f'tr_{t}'
            if key in x:
                x[key] = 1.0
        if t != baseline_label:
            name = f'tr_{t}:day'
            if name in x:
                x[name] = day_ref
            for fn in feat_names:
                if fn.startswith(f'tr_{t}:day'):
                    if fn == f'tr_{t}:day':
                        x[fn] = day_ref
                    else:
                        suffix = fn.split(':', 1)[1]
                        if suffix.startswith('day') and suffix[3:].isdigit():
                            k = int(suffix[3:])
                            x[fn] = day_ref ** k
        x_vec = np.array([x[name] for name in feat_names], dtype=float).reshape(1, -1)
        y_hat = rf_result['model'].predict(x_vec)[0]
        rows.append({
            'treatment': t,
            'reference_day': day_ref,
            'adjusted_mean': float(y_hat),
            'p_value': np.nan,  # RF doesn't provide p-values
            'significant': False,
        })
    return pd.DataFrame(rows).sort_values('adjusted_mean', ascending=False)


def rf_time_cv_search(
    X: np.ndarray,
    y: np.ndarray,
    columns: list[str],
    dates: pd.Series,
    n_splits: int = 3,
    n_iter: int = 10,
    random_state: int = 42,
):
    if not SKLEARN_AVAILABLE:
        return None
    rng = np.random.default_rng(random_state)
    cand_n_estimators = [200, 400, 600, 800]
    cand_max_depth = [None, 4, 6, 8, 10]
    cand_min_samples_split = [2, 4, 6]
    cand_min_samples_leaf = [1, 2, 3]
    # Valid max_features for RandomForestRegressor: {'sqrt','log2', None} or int/float
    cand_max_features = ['sqrt', 'log2', None, 0.5, 0.8]

    def time_splits(n):
        indices = np.argsort(dates.to_numpy().astype('datetime64[ns]'))
        sizes = np.linspace(0.5, 0.8, n_splits)
        for s in sizes:
            n_train = max(int(np.floor(s * n)), 1)
            if n_train >= n:
                continue
            train_idx = indices[:n_train]
            test_idx = indices[n_train:]
            yield train_idx, test_idx

    best = None
    n = len(y)
    for _ in range(n_iter):
        params = {
            'n_estimators': int(rng.choice(cand_n_estimators)),
            'max_depth': rng.choice(cand_max_depth),
            'min_samples_split': int(rng.choice(cand_min_samples_split)),
            'min_samples_leaf': int(rng.choice(cand_min_samples_leaf)),
            'max_features': rng.choice(cand_max_features),
            'random_state': random_state,
            'n_jobs': -1,
        }
        r2_list, mae_list = [], []
        for tr_idx, te_idx in time_splits(n):
            model = RandomForestRegressor(**params)
            model.fit(X[tr_idx], y[tr_idx])
            y_pred = model.predict(X[te_idx])
            r2_list.append(r2_score(y[te_idx], y_pred))
            mae_list.append(mean_absolute_error(y[te_idx], y_pred))
        score = float(np.nanmean(r2_list))
        mae = float(np.nanmean(mae_list))
        if best is None or score > best['r2_cv']:
            best = {'params': params, 'r2_cv': score, 'mae_cv': mae}
    return best


def analyze_treatments(csv_file: str, out_dir: str = 'evaluation_results', min_count: int = 2,
                       model: str = 'ols', test_fraction: float = 0.3, random_state: int = 42,
                       n_estimators: int = 400, max_depth: int | None = None,
                       poly_day: int = 0, interact_day: bool = False,
                       rf_cv: bool = False, rf_cv_iter: int = 10, rf_cv_splits: int = 3,
                       ref_day: float | None = None):
    df = pd.read_csv(csv_file)
    df = standardize_columns(df)
    df = parse_dates(df)

    # Basic filtering
    keep_cols = [c for c in ['date', 'pef', 'fev_1', 'note'] if c in df.columns]
    df = df[keep_cols].copy()

    df['pef'] = to_numeric(df['pef'], remove_suffix=' L/min') if 'pef' in df.columns else np.nan
    if 'fev_1' in df.columns:
        df['fev_1'] = to_numeric(df['fev_1'], remove_suffix=' L')

    # Drop rows without PEF
    df = df.dropna(subset=['pef'])

    df = normalize_treatment_from_note(df)

    # Descriptive stats per treatment (exact label)
    desc_cols = ['pef'] + (['fev_1'] if 'fev_1' in df.columns else [])
    desc = (
        df.groupby('treatment')[desc_cols]
          .agg(['count', 'mean', 'median', 'std'])
    )
    # Flatten columns
    desc.columns = ['_'.join([c for c in col if c]) for col in desc.columns.to_flat_index()]
    desc = desc.reset_index().sort_values('pef_mean', ascending=False)

    # Regression-adjusted effects controlling for day trend
    # PEF model
    pef_adj_df = None
    pef_result = None
    pef_cols = None
    pef_baseline = None
    pef_rf = None
    pef_rf_adj_df = None
    X, y, cols, baseline, design_df = build_design(df, 'pef', min_count_per_treat=min_count, poly_day=poly_day, interact_day=interact_day)
    if X is not None:
        # Choose reference day: user override or median of observed days
        day_ref = ref_day if ref_day is not None else (float(np.median(X[:, cols.index('day')])) if 'day' in cols else 0.0)
        if model in ('ols', 'both'):
            pef_result = ols_fit(X, y)
            pef_cols = cols
            pef_baseline = baseline
            pef_adj_df = adjusted_means(pef_result, pef_cols, pef_baseline, day_ref)
        if model in ('rf', 'both') and SKLEARN_AVAILABLE:
            pef_rf = rf_fit_with_time_split(X, y, cols, design_df['date'], test_fraction, random_state, n_estimators, max_depth)
            treatments_post_group = design_df['treatment'].value_counts().index.tolist()
            pef_rf_adj_df = rf_adjusted_means(pef_rf, cols, baseline, treatments_post_group, day_ref)
            pef_rf_cv = rf_time_cv_search(X, y, cols, design_df['date'], n_splits=rf_cv_splits, n_iter=rf_cv_iter, random_state=random_state) if rf_cv else None

    # FEV-1 model
    fev1_adj_df = None
    fev1_result = None
    fev1_cols = None
    fev1_baseline = None
    fev1_rf = None
    fev1_rf_adj_df = None
    if 'fev_1' in df.columns and df['fev_1'].notna().any():
        X2, y2, cols2, baseline2, design_df2 = build_design(
            df, 'fev_1', min_count_per_treat=min_count, poly_day=poly_day, interact_day=interact_day
        )
        if X2 is not None:
            day_ref2 = ref_day if ref_day is not None else (float(np.median(X2[:, cols2.index('day')])) if 'day' in cols2 else 0.0)
            if model in ('ols', 'both'):
                fev1_result = ols_fit(X2, y2)
                fev1_cols = cols2
                fev1_baseline = baseline2
                fev1_adj_df = adjusted_means(fev1_result, fev1_cols, fev1_baseline, day_ref2)
            if model in ('rf', 'both') and SKLEARN_AVAILABLE:
                fev1_rf = rf_fit_with_time_split(X2, y2, cols2, design_df2['date'], test_fraction, random_state, n_estimators, max_depth)
                treatments_post_group2 = design_df2['treatment'].value_counts().index.tolist()
                fev1_rf_adj_df = rf_adjusted_means(fev1_rf, cols2, baseline2, treatments_post_group2, day_ref2)
                fev1_rf_cv = rf_time_cv_search(X2, y2, cols2, design_df2['date'], n_splits=rf_cv_splits, n_iter=rf_cv_iter, random_state=random_state) if rf_cv else None

    # Prepare outputs
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(out_dir, exist_ok=True)

    base_name = f"treatment_analysis_{ts}"
    desc_csv = os.path.join(out_dir, base_name + '_descriptive.csv')
    desc.to_csv(desc_csv, index=False)

    result_json = {
        'input_file': csv_file,
        'generated_at': ts,
        'n_rows': int(len(df)),
        'unique_treatments': int(df['treatment'].nunique()),
        'descriptive_top5': desc.head(5).to_dict(orient='records'),
        'pef_model_ols': {
            'baseline': pef_baseline,
            'r2': None if pef_result is None else float(pef_result['r2']),
            'dof': None if pef_result is None else int(pef_result['dof']),
        },
        'fev1_model_ols': {
            'baseline': fev1_baseline,
            'r2': None if fev1_result is None else float(fev1_result['r2']),
            'dof': None if fev1_result is None else int(fev1_result['dof']),
        },
    }

    if model in ('rf', 'both'):
        result_json.update({
            'pef_model_rf': None if pef_rf is None else {
                'r2_test': pef_rf['r2_test'],
                'mae_test': pef_rf['mae_test'],
                'n_train': pef_rf['n_train'],
                'n_test': pef_rf['n_test'],
                'cv': None if 'pef_rf_cv' not in locals() else pef_rf_cv,
            },
            'fev1_model_rf': None if fev1_rf is None else {
                'r2_test': fev1_rf['r2_test'],
                'mae_test': fev1_rf['mae_test'],
                'n_train': fev1_rf['n_train'],
                'n_test': fev1_rf['n_test'],
                'cv': None if 'fev1_rf_cv' not in locals() else fev1_rf_cv,
            }
        })

    json_path = os.path.join(out_dir, base_name + '.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result_json, f, indent=2)

    # Save adjusted means if available
    if pef_adj_df is not None:
        pef_csv = os.path.join(out_dir, base_name + '_pef_adjusted_means.csv')
        pef_adj_df.to_csv(pef_csv, index=False)

    if fev1_adj_df is not None:
        fev1_csv = os.path.join(out_dir, base_name + '_fev1_adjusted_means.csv')
        fev1_adj_df.to_csv(fev1_csv, index=False)

    # Save RF adjusted means
    pef_rf_csv = None
    fev1_rf_csv = None
    if pef_rf_adj_df is not None:
        pef_rf_csv = os.path.join(out_dir, base_name + '_pef_rf_adjusted_means.csv')
        pef_rf_adj_df.to_csv(pef_rf_csv, index=False)
    if fev1_rf_adj_df is not None:
        fev1_rf_csv = os.path.join(out_dir, base_name + '_fev1_rf_adjusted_means.csv')
        fev1_rf_adj_df.to_csv(fev1_rf_csv, index=False)

    # Build outputs dict early for return
    outputs_dict = {
        'json': json_path,
        'desc_csv': desc_csv,
        'pef_csv': pef_csv if pef_adj_df is not None else None,
        'fev1_csv': fev1_csv if fev1_adj_df is not None else None,
        'pef_rf_csv': pef_rf_csv,
        'fev1_rf_csv': fev1_rf_csv,
    }

    # Console summary
    print('\n=== Descriptive (by treatment) ===')
    print(desc[['treatment'] + [c for c in desc.columns if c != 'treatment']].head(10).to_string(index=False))

    if pef_adj_df is not None:
        print('\n=== Adjusted PEF (higher is better) ===')
        print(pef_adj_df.head(10).to_string(index=False))

    if fev1_adj_df is not None:
        print('\n=== Adjusted FEV-1 (higher is better) ===')
        print(fev1_adj_df.head(10).to_string(index=False))

    if model in ('rf', 'both'):
        if pef_rf is not None:
            print('\n=== Random Forest PEF (time-split metrics) ===')
            print(f"R^2_test={pef_rf['r2_test']}, MAE_test={pef_rf['mae_test']} (n_train={pef_rf['n_train']}, n_test={pef_rf['n_test']})")
            if 'pef_rf_cv' in locals() and pef_rf_cv is not None:
                print('CV best (PEF):', pef_rf_cv)
        if pef_rf_adj_df is not None:
            print('\n=== RF Adjusted PEF (higher is better) ===')
            print(pef_rf_adj_df.head(10).to_string(index=False))
        if fev1_rf is not None:
            print('\n=== Random Forest FEV-1 (time-split metrics) ===')
            print(f"R^2_test={fev1_rf['r2_test']}, MAE_test={fev1_rf['mae_test']} (n_train={fev1_rf['n_train']}, n_test={fev1_rf['n_test']})")
            if 'fev1_rf_cv' in locals() and fev1_rf_cv is not None:
                print('CV best (FEV-1):', fev1_rf_cv)
        if fev1_rf_adj_df is not None:
            print('\n=== RF Adjusted FEV-1 (higher is better) ===')
            print(fev1_rf_adj_df.head(10).to_string(index=False))

    return {
        'desc': desc,
        'pef_adj': pef_adj_df,
        'fev1_adj': fev1_adj_df,
        'pef_rf_adj': pef_rf_adj_df,
        'fev1_rf_adj': fev1_rf_adj_df,
        'outputs': outputs_dict,
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze treatments for best PEF/FEV-1 results')
    parser.add_argument('csv_file', nargs='?', default=r'data_files/pef_data_20251215.csv', help='Path to input CSV')
    parser.add_argument('--min-count', type=int, default=5, help='Minimum records per treatment to include as its own category (others grouped into "Other")')
    parser.add_argument('--model', choices=['ols', 'rf', 'both'], default='ols', help='Model to use for adjusted comparisons')
    parser.add_argument('--test-fraction', type=float, default=0.3, help='Fraction of data used for time-aware test split')
    parser.add_argument('--n-estimators', type=int, default=400, help='Number of trees for Random Forest')
    parser.add_argument('--max-depth', type=int, default=None, help='Max depth for Random Forest (None for unlimited)')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--poly-day', type=int, default=0, help='Add polynomial day terms up to this degree (e.g., 2 adds day^2)')
    parser.add_argument('--interact-day', action='store_true', help='Include treatment × day interaction terms (and with polynomial terms if specified)')
    parser.add_argument('--rf-cv', action='store_true', help='Run time-series CV randomized search for RF and report best params and CV metrics')
    parser.add_argument('--rf-cv-iter', type=int, default=10, help='Number of random parameter samples for RF CV')
    parser.add_argument('--rf-cv-splits', type=int, default=3, help='Number of forward-chaining CV splits')
    parser.add_argument('--ref-day', type=float, default=None, help='Override reference day (if omitted, uses median of observed days)')
    parser.add_argument('--plot', action='store_true', help='Display publication-quality treatment effect chart with 95%% CIs and significance markers')
    args = parser.parse_args()

    if args.model in ('rf', 'both') and not SKLEARN_AVAILABLE:
        print('Error: scikit-learn is not installed. Please install scikit-learn to use Random Forest.')
        return

    res = analyze_treatments(
        args.csv_file,
        min_count=args.min_count,
        model=args.model,
        test_fraction=args.test_fraction,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        poly_day=args.poly_day,
        interact_day=args.interact_day,
        rf_cv=args.rf_cv,
        rf_cv_iter=args.rf_cv_iter,
        rf_cv_splits=args.rf_cv_splits,
        ref_day=args.ref_day,
    )
    print('\nSaved outputs:')
    for k, v in res['outputs'].items():
        if v:
            print(f" - {k}: {v}")

    if args.plot:
        json_path = res['outputs'].get('json') or 'evaluation_results'
        plot_treatment_effects(res, out_dir=os.path.dirname(json_path) or 'evaluation_results')


def plot_treatment_effects(res: dict, out_dir: str = 'evaluation_results', show: bool = True):
    """Generate a publication-quality forest-style plot of treatment effects.

    Uses diamond markers, 95% confidence intervals as horizontal error bars,
    and asterisks to denote statistical significance (p < 0.05).
    """
    import matplotlib.pyplot as plt

    pef_adj = res.get('pef_adj')
    fev1_adj = res.get('fev1_adj')

    figs_created = []

    def _make_plot(df: pd.DataFrame, metric_name: str, units: str):
        if df is None or df.empty:
            return None
        df = df.copy().sort_values('adjusted_mean', ascending=True)

        fig, ax = plt.subplots(figsize=(8, max(4, 0.5 * len(df))))

        y_pos = np.arange(len(df))
        means = df['adjusted_mean'].to_numpy()
        labels = df['treatment'].tolist()

        # Compute error bar sizes (distance from mean to CI bounds)
        if 'ci_lower' in df.columns and 'ci_upper' in df.columns:
            err_lower = means - df['ci_lower'].to_numpy()
            err_upper = df['ci_upper'].to_numpy() - means
            xerr = np.vstack([err_lower, err_upper])
        else:
            xerr = None

        # Colour by significance
        if 'significant' in df.columns:
            colors = ['#2ca02c' if sig else '#1f77b4' for sig in df['significant']]
        else:
            colors = '#1f77b4'

        ax.errorbar(
            means, y_pos, xerr=xerr, fmt='none', ecolor='gray',
            elinewidth=1.5, capsize=3, capthick=1.5, zorder=1,
        )
        ax.scatter(
            means, y_pos, marker='D', s=80, c=colors, edgecolors='black',
            linewidths=0.8, zorder=2,
        )

        # Add significance asterisks
        if 'p_value' in df.columns:
            for i, (m, p) in enumerate(zip(means, df['p_value'])):
                if pd.notna(p):
                    if p < 0.001:
                        sig_label = '***'
                    elif p < 0.01:
                        sig_label = '**'
                    elif p < 0.05:
                        sig_label = '*'
                    else:
                        sig_label = ''
                    if sig_label:
                        ax.text(m, i + 0.25, sig_label, ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel(f'Adjusted {metric_name} ({units})', fontsize=11)
        ax.set_title(f'Treatment Effects on {metric_name} (95% CI)', fontsize=13, fontweight='bold')
        ax.axvline(means[df['treatment'] == df['treatment'].iloc[0]].mean(), color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.grid(axis='x', linestyle=':', alpha=0.5)

        # Add legend for significance
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='D', color='w', markerfacecolor='#2ca02c', markersize=10,
                   markeredgecolor='black', label='Significant (p<0.05)'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='#1f77b4', markersize=10,
                   markeredgecolor='black', label='Not significant'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9)

        plt.tight_layout()
        return fig

    if pef_adj is not None and not pef_adj.empty:
        fig_pef = _make_plot(pef_adj, 'PEF', 'L/min')
        if fig_pef:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            path_pef = os.path.join(out_dir, f'treatment_effects_pef_{ts}.png')
            fig_pef.savefig(path_pef, dpi=300, bbox_inches='tight')
            figs_created.append(('pef_plot', path_pef))
            print(f"Saved PEF treatment effect plot: {path_pef}")

    if fev1_adj is not None and not fev1_adj.empty:
        fig_fev1 = _make_plot(fev1_adj, 'FEV-1', 'L')
        if fig_fev1:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            path_fev1 = os.path.join(out_dir, f'treatment_effects_fev1_{ts}.png')
            fig_fev1.savefig(path_fev1, dpi=300, bbox_inches='tight')
            figs_created.append(('fev1_plot', path_fev1))
            print(f"Saved FEV-1 treatment effect plot: {path_fev1}")

    if show:
        plt.show()

    return figs_created


if __name__ == '__main__':
    main()
