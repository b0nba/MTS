from scipy.stats import chi2

def get_chi_square_threshold(feature_count, alpha=0.05):
    return chi2.ppf(
        1 - alpha,
        df=feature_count
    ) / feature_count