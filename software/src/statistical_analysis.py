import pandas as pd
import numpy as np
import scipy
import yaml 

from scipy.stats import chi2_contingency
from file_management import save_expected_table

# Load configuration from YAML file
try:
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
except FileNotFoundError:
    print("Configuration file not found. Please check the file path.")
    exit(1)
except yaml.YAMLError as exc:
    print(f"Error in configuration file: {exc}")
    exit(1)

# Load constants & hyperparameters from YAML config
not_applicable = config['constants']['not_applicable']
chi2_squared_test = config['constants']['chi2_squared_test']
fisher_exact_test = config['constants']['fisher_exact_test']
report_critical_p_value = config['hyperparameters']['report_critical_p_value']



def get_chi2_or_fisher_p_value_and_OR(contingency_table, df_name):
    """
    Calculates the p-value using the Chi-squared test or Fisher's Exact test based on the suitability for the provided contingency table.

    This function first attempts to compute the p-value using the Chi-squared test. 
    If the Chi-squared test is not applicable due to the expected table (derived from the contingency table) contains a zero cell or less than 5 (zero will through an error), 
    the function falls back to Fisher's Exact test.

    Parameters:
    - contingency_table (pd.DataFrame): A 2x2 contingency table.

    Returns:
    - tuple: A tuple containing the p-value and a string indicating which test was used 
    """
    vote_vote = contingency_table.loc['V','V'] + 0.5
    vote_noVote = contingency_table.loc['V','¬V'] + 0.5
    noVote_vote = contingency_table.loc['¬V','V'] + 0.5
    noVote_noVote = contingency_table.loc['¬V','¬V'] + 0.5

    OR_value = not_applicable

    try:
        stat, chi2_p, dof, expected_table = chi2_contingency(contingency_table, correction= False)
        expected_df = pd.DataFrame(expected_table, 
                            index=contingency_table.index, 
                            columns=contingency_table.columns).astype(int)
        save_expected_table(expected_df, df_name)
        if (expected_table < 5).any():
            _, fisher_exact_p = scipy.stats.fisher_exact(contingency_table)
            if (fisher_exact_p < report_critical_p_value and (vote_noVote * noVote_vote) > 0):
                OR_value = (vote_vote * noVote_noVote) / (vote_noVote * noVote_vote)
            return (fisher_exact_p, fisher_exact_test, OR_value)
        if (chi2_p < report_critical_p_value and (vote_noVote * noVote_vote) > 0):
            OR_value = (vote_vote * noVote_noVote) / (vote_noVote * noVote_vote)
    except ValueError as e:
        _, fisher_exact_p = scipy.stats.fisher_exact(contingency_table)
        if (fisher_exact_p < report_critical_p_value and (vote_noVote * noVote_vote) > 0):
            OR_value = (vote_vote * noVote_noVote) / (vote_noVote * noVote_vote)
        return (fisher_exact_p, fisher_exact_test, OR_value)
    return (chi2_p, chi2_squared_test, OR_value)

def get_yuleQ_value(contingency_table, isSkip):
    """
    Calculates the Yule's Q coefficient for a given 2x2 contingency table, unless skipping
    
    Yule'sQ value ranges from:
        -1 (perfect negative association) to 
        +1 (perfect positive association), 
        with 0 indicating no association. T
    This function computes Yule's Q only if all cells in the contingency table have non-zero values. 
    It skips calculation if the same sense is being compared (i=j) or if any cell in the table is zero to avoid division by zero.

    Parameters:
    - contingency_table (pd.DataFrame): A 2x2 contingency table.
    - isSkip (bool): A flag indicating whether to skip the calculation. 

    Returns:
    - float or str: Yule's Q coefficient if calculable, otherwise 'NA' if skipped or if the table contains zero values.
    """
    vote_vote = contingency_table.loc['V','V']
    vote_noVote = contingency_table.loc['V','¬V']
    noVote_vote = contingency_table.loc['¬V','V']
    noVote_noVote = contingency_table.loc['¬V','¬V']

    if (isSkip):
        # we set isSkip to true when we are comparing the same sense (i=j)
        return not_applicable
    if (vote_vote*vote_noVote*noVote_vote*noVote_noVote == 0):  
        # none of the contingency table cells can be zero
        return not_applicable
    numerator = (vote_vote * noVote_noVote) - (vote_noVote * noVote_vote)
    denominator = (vote_vote * noVote_noVote) + (vote_noVote * noVote_vote)
    yules_q = numerator / denominator

    return yules_q

def get_proposed_method_value(contingency_table):
    vote_vote = contingency_table.loc['V','V']
    vote_noVote = contingency_table.loc['V','¬V']
    noVote_vote = contingency_table.loc['¬V','V']

    numerator = vote_vote
    denominator = vote_vote + vote_noVote + noVote_vote

    if denominator == 0:
        return not_applicable  

    proposed_method_value = numerator/denominator
    return proposed_method_value

def get_conditional_probability_value(contingency_table):
    vote_vote = contingency_table.loc['V','V']
    vote_noVote = contingency_table.loc['V','¬V']
    noVote_vote = contingency_table.loc['¬V','V']

    s1_given_s2_probability = None
    s2_given_s1_probability = None
    
    if (vote_vote + noVote_vote) > 0:
        s1_given_s2_probability = vote_vote / (vote_vote + noVote_vote)
    if (vote_vote + vote_noVote) > 0:
        s2_given_s1_probability = vote_vote / (vote_vote + vote_noVote)
    
    if s1_given_s2_probability is not None and s2_given_s1_probability is not None:
        return (s1_given_s2_probability + s2_given_s1_probability) / 2
    elif s1_given_s2_probability is not None:
        return s1_given_s2_probability
    elif s2_given_s1_probability is not None:
        return s2_given_s1_probability
    else:
        return not_applicable
    
def get_pointwise_mutual_info_value(contingency_table):
    vote_vote = contingency_table.loc['V','V']
    vote_noVote = contingency_table.loc['V','¬V']
    noVote_vote = contingency_table.loc['¬V','V']
    noVote_noVote = contingency_table.loc['¬V','¬V']

    total = vote_vote + vote_noVote + noVote_vote + noVote_noVote
    p_v_v = vote_vote / total
    p_v_s1 = (vote_vote + vote_noVote) / total
    p_v_s2 = (vote_vote + noVote_vote) / total

    if (p_v_v > 0 and (p_v_s1 * p_v_s2) > 0):
        pointwise_mutual_info_value = np.log2(p_v_v / (p_v_s1 * p_v_s2))
    else:
        pointwise_mutual_info_value = not_applicable

    return pointwise_mutual_info_value