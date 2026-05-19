import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm


def vase_filter(df, hw_filt, width_filt, qauntile):
    grouped = df.groupby('gpp_no')
    # Profiles are right-half only (x >= 0), so full width = 2 * x.max()
    full_width = 2 * grouped['x'].max()
    height = grouped['y'].max() - grouped['y'].min()

    h_w_ratio = height / full_width
    to_remove_hw = h_w_ratio[h_w_ratio < hw_filt].index

    top_quant_thresh = df['y'].quantile(1 - qauntile)
    top_width = df[df['y'] >= top_quant_thresh]['x'].pipe(lambda s: s.max() - s.min())
    bot_width = df[df['y'] <= df['y'].quantile(qauntile)]['x'].pipe(lambda s: s.max() - s.min())
    to_remove_width = df['gpp_no'].unique() if (bot_width / top_width) < width_filt else []

    to_keep = set(df['gpp_no']) - set(to_remove_hw) - set(to_remove_width)
    print(f"Removed {df['gpp_no'].nunique() - len(to_keep)} vases.\n"
          f"{len(to_remove_hw)} by height-to-width ratio and {len(to_remove_width)} by width ratio.")

    return df[df['gpp_no'].isin(to_keep)], df[~df['gpp_no'].isin(to_keep)]['gpp_no'].unique()


def pca_run(df, hw_filt, width_filt=0, quantile=0.05):
    df, gone_list = vase_filter(df, hw_filt, width_filt, quantile)

    merged_data_trim = df.drop_duplicates(subset=['gpp_no', 'point_order'])

    x = merged_data_trim.pivot(index='gpp_no', columns='point_order', values='x').reset_index()
    y = merged_data_trim.pivot(index='gpp_no', columns='point_order', values='y').reset_index()
    d = pd.concat([x, y.drop(columns=['gpp_no'])], axis=1)
    d.columns = ['gpp_no'] + [f'coord_{i}' for i in range(1, d.shape[1])]

    pca = PCA(n_components=6)
    pca_result = pca.fit_transform(d.drop(columns=['gpp_no']).values)

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = explained_variance.cumsum()
    print(len(explained_variance))
    print(cumulative_variance[:10].round(2))

    pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])
    pca_df['gpp_no'] = d['gpp_no'].reset_index(drop=True)
    culture_map = merged_data_trim.set_index('gpp_no')['culture_general'].to_dict()
    pca_df['culture_general'] = [
        culture_map.get(g, None) for g in tqdm(pca_df['gpp_no'], desc="Adding culture_general")
    ]

    pc_vals_df = pd.DataFrame(pca.components_, columns=d.columns[1:])

    return pca_df, explained_variance, cumulative_variance, pc_vals_df, pca.mean_, pca_result, pca, d, gone_list


def pca_project(df, out_dir, hw_filt=0, width_filt=0, quantile=0.05):
    """Project pot profiles onto a previously saved PCA space."""
    df, gone_list = vase_filter(df, hw_filt, width_filt, quantile)

    merged_data_trim = df.drop_duplicates(subset=['gpp_no', 'point_order'])

    x = merged_data_trim.pivot(index='gpp_no', columns='point_order', values='x').reset_index()
    y = merged_data_trim.pivot(index='gpp_no', columns='point_order', values='y').reset_index()
    d = pd.concat([x, y.drop(columns=['gpp_no'])], axis=1)
    d.columns = ['gpp_no'] + [f'coord_{i}' for i in range(1, d.shape[1])]

    means = pd.read_csv(f'{out_dir}/pc_val_means.csv')['pc_mean'].values
    components_df = pd.read_csv(f'{out_dir}/pc_vals.csv', index_col=0)
    components = components_df.values

    X = d.drop(columns=['gpp_no']).values

    if X.shape[1] != len(means):
        raise ValueError(
            f"Feature dimension mismatch: current data has {X.shape[1]} features "
            f"but saved PCA expects {len(means)}.\n"
            f"  point_order range in current data: "
            f"{int(merged_data_trim['point_order'].min())}–{int(merged_data_trim['point_order'].max())} "
            f"({merged_data_trim['point_order'].nunique()} unique values)\n"
            f"  Expected feature columns (from pc_vals.csv): {components_df.shape[1]}\n"
            f"Ensure the profile parquet file matches the one used for the original PCA."
        )

    scores = (X - means) @ components.T

    pca_df = pd.DataFrame(scores, columns=[f'PC{i+1}' for i in range(scores.shape[1])])
    pca_df['gpp_no'] = d['gpp_no'].values
    culture_map = merged_data_trim.set_index('gpp_no')['culture_general'].to_dict()
    pca_df['culture_general'] = [
        culture_map.get(g, None) for g in tqdm(pca_df['gpp_no'], desc="Adding culture_general")
    ]

    return pca_df, d, gone_list
