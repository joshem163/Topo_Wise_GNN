from scipy.stats import ttest_rel
def statistical_sig(model1_scores,model2_scores):
    t_statistic, p_value = ttest_rel(model1_scores, model2_scores)
    if p_value < 0.001:
        significance_stars='***'  # Very highly significant
    elif p_value < 0.01:
        significance_stars='**'  # Highly significant
    elif p_value < 0.05:
        significance_stars='*'  # Significant
    else:
        significance_stars='Not significant'
    print(f"P-value: {p_value}, Significance: {significance_stars}")