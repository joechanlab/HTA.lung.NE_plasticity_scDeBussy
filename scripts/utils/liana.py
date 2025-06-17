import pandas as pd
from scipy.stats import rankdata
import numpy as np
from ._aggregate import _robust_rank_aggregate 
# download to utils/ from liana repo: https://github.com/saezlab/liana-py/blob/e0c86b15cc2731dde8f4caf63e7918c032f4f2b3/liana/method/_pipe_utils/_aggregate.py

def directional_rank(df: pd.DataFrame, score_columns: list[str], ascending_flags: list[bool]) -> pd.DataFrame:
    """
    Rank columns using scipy's rankdata to match LIANA's internal ranking logic.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the score columns to rank.
    score_columns : list[str]
        List of column names to rank.
    ascending_flags : list[bool]
        List of flags indicating whether each column should be ranked in ascending order.

    Returns
    -------
    pd.DataFrame
        DataFrame with ranked columns (same shape as input).
    """
    ranked_dict = {}

    for col, asc in zip(score_columns, ascending_flags):
        data = df[col].values
        # If higher is better, reverse the data by multiplying by -1
        data_to_rank = data if asc else -1 * data
        ranked_dict[col] = rankdata(data_to_rank, method="average")
    
    return pd.DataFrame(ranked_dict, index=df.index)

if __name__ == "__main__":
    test_df = pd.read_csv('https://github.com/saezlab/liana-py/raw/e0c86b15cc2731dde8f4caf63e7918c032f4f2b3/liana/tests/data/aggregate_rank_rest.csv')
    # For specificity scores — lower = better (e.g. pvals) → ascending rank
    spec_cols = ['cellphone_pvals',  'spec_weight', 'scaled_weight', 'lr_logfc']
    spec_ascending = [True, False, False, False]  # `lr_logfc` and `spec_weight`: higher = better
    spec_rmat = directional_rank(test_df, spec_cols, spec_ascending)
    spec_pvals = _robust_rank_aggregate(spec_rmat.to_numpy())
    # Compare specificity ranks
    print("\nSpecificity rank comparison:")
    print("Original vs Computed correlation:", np.corrcoef(test_df["specificity_rank"], spec_pvals)[0,1])
    print("\nFirst 5 rows comparison:")
    comparison_df = pd.DataFrame({
        'original': test_df["specificity_rank"].head(5),
        'computed': spec_pvals[:5]
    })
    print(comparison_df)

    # Check for almost exact matches
    is_close = np.isclose(test_df["specificity_rank"], spec_pvals, rtol=1e-10, atol=1e-10)
    print("\nAlmost exact matches (within 1e-10):")
    print(f"Number of matches: {np.sum(is_close)} out of {len(is_close)}")
    print(f"Percentage match: {np.mean(is_close)*100:.2f}%")

    if not np.all(is_close):
        print("\nFirst few mismatches:")
        mismatch_idx = np.where(~is_close)[0][:5]
        mismatch_df = pd.DataFrame({
            'original': test_df["specificity_rank"].iloc[mismatch_idx],
            'computed': spec_pvals[mismatch_idx],
            'diff': np.abs(test_df["specificity_rank"].iloc[mismatch_idx] - spec_pvals[mismatch_idx])
        })
        print(mismatch_df)

    # For magnitude scores — higher = better → descending rank
    mag_cols = ['lr_means', 'expr_prod', 'lrscore']
    mag_ascending = [False, True, False]
    mag_rmat = directional_rank(test_df, mag_cols, mag_ascending)

    # Pass raw ranks directly to _robust_rank_aggregate
    mag_pvals = _robust_rank_aggregate(mag_rmat.to_numpy())

    spec_cols = ['cellphone_pvals',  'spec_weight', 'scaled_weight', 'lr_logfc']
    spec_ascending = [True, False, False, False]  # `lr_logfc` and `spec_weight`: higher = better
    spec_rmat = directional_rank(test_df, spec_cols, spec_ascending)
    spec_pvals = _robust_rank_aggregate(spec_rmat.to_numpy())
    # Compare specificity ranks
    print("\nMagnitude rank comparison:")
    print("Original vs Computed correlation:", np.corrcoef(test_df["magnitude_rank"], mag_pvals)[0,1])
    print("\nFirst 5 rows comparison:")
    comparison_df = pd.DataFrame({
        'original': test_df["magnitude_rank"].head(5),
        'computed': mag_pvals[:5]
    })
    print(comparison_df)

    # Check for almost exact matches
    is_close = np.isclose(test_df["magnitude_rank"], mag_pvals, rtol=1e-10, atol=1e-10)
    print("\nAlmost exact matches (within 1e-10):")
    print(f"Number of matches: {np.sum(is_close)} out of {len(is_close)}")
    print(f"Percentage match: {np.mean(is_close)*100:.2f}%")

    if not np.all(is_close):
        print("\nFirst few mismatches:")
        mismatch_idx = np.where(~is_close)[0][:5]
        mismatch_df = pd.DataFrame({
            'original': test_df["magnitude_rank"].iloc[mismatch_idx],
            'computed': mag_pvals[mismatch_idx],
            'diff': np.abs(test_df["magnitude_rank"].iloc[mismatch_idx] - mag_pvals[mismatch_idx])
        })
        print(mismatch_df)