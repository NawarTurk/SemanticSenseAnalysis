import pandas as pd
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import numpy as np
import glob
import os
from scipy.stats import chi2_contingency
from scipy.stats import fisher_exact

# Dataframe dictionaries
dfs_ready_to_transform = {}
dfs_ready_to_process = {}
dfs_ready_to_process_continuous = {}  # not active currently 

global_sheet_df = pd.DataFrame()

# Value matrices dictionaries     
chi2_p_value_matrices = {}
fisher_exact_p_value_matrices = {}
OR_ratio_matrices = {}
yuleQ_value_matrices = {}
proposed_indicator_value_matrices = {}
conditional_probability_value_matrices = {}
pointwise_mutual_info_value_matrices = {}


# Folder & file names
raw_data_folder = '0_raw_data'
ready_to_transform_folder = '1_ready_to_transform'
ready_to_process_folder = '2_ready_to_process'
result_folder = '3_results'
binary_folder = 'binary'
continuous_folder = 'continuous'
analysis_value_matrices_folder = 'analysis_value_matrices'
heatmap_folder = 'heatmaps'
csv_files_folder = 'csv_files'
contingency_table_folder = 'contingency_tables'
expected_table_folder = 'expected_tables'
report_folder = 'report'
summary_report_folder = 'summary_report'

# Others
not_applicable = 'NA'
fisher_exact_test ='fisher'
chi2_squared_test = 'chi2'
yule_Q_test = 'yuleQ'
proposed_indicator = 'proposed_indicator'
conditional_probability = 'conditional_probability'
OR_ratio = 'OR'
pointwise_mutual_info = 'pointwise_mutual_info_mutual_info'



#  *** DATA PROCESSING FUNCTIONS ***
def transform_to_binary(df_ready_to_transform, threshold):
    """
    Converts each value in the input DataFrame to binary (1 or 0) based on a specified threshold

    Parameters:
    - df_ready_to_transform (pandas.DataFrame): DataFrame to transform.
    - threshold (numeric): Value serving as the cutoff point for binary conversion; values equal to or above the threshold are set to 1, others to 0.

    Returns:
    - pandas.DataFrame: Transformed DataFrame with binary values.
    """
    return df_ready_to_transform.map(lambda x: 'V' if x >= threshold else '¬V')

def remove_all_zeros_columns(df_transformed_to_binary):
    all_zero_columns = df_transformed_to_binary.columns[(df_transformed_to_binary == '¬V').all()]
    df_binary_cleaned = df_transformed_to_binary.drop(all_zero_columns, axis=1)
    return df_binary_cleaned

def get_contingency_matrix(s1, s2, s1_name, s2_name, df_name, isSkip):
    """
    Creates a contingency table comparing of two senses.
    
    If not skipped, the table is also appended to a text file named after the `df_name` parameter within a structured directory path.
    
    Parameters:
    - s1 (pandas.Series): First sense to compare.
    - s2 (pandas.Series): Second sense to compare.
    - s1_name (str): Name to assign to the rows in the contingency table.
    - s2_name (str): Name to assign to the columns in the contingency table.
    - df_name (str): Base name for the output text file (used if `isSkip` is False).
    - isSkip (bool): Flag to skip file writing if True. (we are effectively skipping when sense 1 and sense 2 are the same sense (i=j))
    
    Returns:
    - pandas.DataFrame: The generated contingency table.
    """
    # s1 = pd.Categorical(s1, categories=[0, 1]) # we had to explictly mention the categories because it will not be added to the table if they a col or a row has values of zeros
    # s2 = pd.Categorical(s2, categories=[0, 1])
    contingency_table = pd.crosstab(s1, s2, rownames=[s1_name], colnames=[s2_name], dropna=False)
    contingency_table_str = str(contingency_table)

    if (not isSkip):
        with open(f'./{result_folder}/{binary_folder}/{contingency_table_folder}/{df_name}.txt', 'a') as file:
            file.write(contingency_table_str + '\n\n' + '-.-.-.-.-.-.-.-.-.-\n')
    return contingency_table

def get_chi2_or_fisher_p_value_and_OR(contingency_table):
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
        save_expected_table(expected_df)
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

#  *** HELPER FUNCTIONS ***
def clean_files_within_directory(directory_name):
    """
    Clean all the files of the previous run to generate new results

    Parameters:
    directory_name (str): The name of the directory, relative to the script's location, to be cleaned
    """
    # Construct the full path to the directory relative to the current script
    directory_path = os.path.join(os.path.dirname(__file__), directory_name)  
    # Walk through the directory
    for root, dirs, files in os.walk(directory_path):
        # Remove each file in the root and subdirectories
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)


def group_to_level2(df):
    """
    Convert the dataframe from leaves level to level 2

    Parameters:
    df (pandas.DataFrame): The leaves level dataframe to be converted

    Returns:
    pandas.DataFrame: The converted dataframe
    """
    
    df_level2 = pd.DataFrame()
    df_level2['synchronous'] = df['synchronous']
    df_level2['asynchronous'] = df['precedence'] + df['succession'] 
    df_level2['cause'] = df['reason'] + df['result'] 
    df_level2['condition'] = df['arg1-as-cond'] + df['arg2-as-cond'] 
    df_level2['negative-condition'] = df['arg1-as-negcond'] + df['arg2-as-negcond'] 
    df_level2['purpose'] = df['arg1-as-goal'] + df['arg2-as-goal'] 
    df_level2['concession'] = df['arg1-as-denier'] + df['arg2-as-denier'] 
    df_level2['contrast'] = df['contrast']
    df_level2['similarity'] = df['similarity']
    df_level2['conjunction'] = df['conjunction']
    df_level2['disjunction'] = df['disjunction']
    df_level2['instantiation'] = df['arg1-as-instance'] + df['arg2-as-instance'] 
    df_level2['level-of-detail'] = df['arg1-as-detail'] + df['arg2-as-detail'] 
    df_level2['equivalence'] = df['equivalence']
    df_level2['manner'] = df['arg1-as-manner'] + df['arg2-as-manner'] 
    df_level2['exception'] = df['arg1-as-excpt'] + df['arg2-as-excpt'] 

    # not both files have the 'arg1-as-subst' column in it, so we have to check
    if 'arg1-as-subst' in df.columns: 
        df_level2['substitution'] = df['arg1-as-subst'] + df['arg2-as-subst'] 
    else:
        df_level2['substitution'] = df['arg2-as-subst'] 
    return df_level2


def generate_csv(matrix_value_df, title):
    matrix_value_df.to_csv(f'./{result_folder}/{binary_folder}/{analysis_value_matrices_folder}/{csv_files_folder}/{title}.csv')

def save_expected_table(expected_table):
    with open(f'./{result_folder}/{binary_folder}/{expected_table_folder}/{df_name}.txt', 'a') as file:
        expected_table_str = str(expected_table)
        file.write(expected_table_str + '\n\n' + '-.-.-.-.-.-.-.-.-.-\n')   

# NEED REFACTORING
def get_global_Sheet_columns():
    golbal_sheet_col_names = []
    for df_name, df_ready_to_transform in dfs_ready_to_process.items():
        for stats_name in statistics_methods:
            col_name = f'{df_name}_{stats_name}'
            golbal_sheet_col_names.append(col_name)
    return golbal_sheet_col_names

def store_in_global_sheet(global_sheet_df, df_name, s1_name, s2_name, p_value, OR_value, yuleQ_value, pointwise_mutual_info_value, proposed_indicator_value):
    if 'leaves' in df_name:
        row_index = f'Leaves: {s1_name}|{s2_name}'
    else:
        row_index = f'Level2: {s1_name}|{s2_name}'

    values_to_store = {
        f'{df_name}_{OR_ratio}': OR_value,
        f'{df_name}_{yule_Q_test}': yuleQ_value,
        f'{df_name}_{pointwise_mutual_info}': pointwise_mutual_info_value,
        f'{df_name}_{proposed_indicator}': proposed_indicator_value
    }

    if row_index not in global_sheet_df.index:
        # Adding a new row by creating a new DataFrame and appending it to the global_sheet_df
        new_row_df = pd.DataFrame(values_to_store, index=[row_index])
        global_sheet_df = pd.concat([global_sheet_df, new_row_df])
    else:
        # If the row already exists, update the values directly
        for col, value in values_to_store.items():
            global_sheet_df.at[row_index, col] = value

    return global_sheet_df


def generate_summary_report(df, df_name, method_used):
    result = ''
    positive_assosication = ''
    negative_assosication = ''

    if method_used == chi2_squared_test:
        result += (f'*** {df_name} ***\n\n')
        for row_label, row in df.iterrows():
            for col_label, value in row.items():
                if row_label != col_label:
                    if isinstance(value, (int, float)):
                        if value < report_critical_p_value:
                            result += (f'{row_label:<20} | {col_label:<20} | {round(value, 4):<7} | \n')
        result += ('______________________________\n\n\n')
        with open(f'{result_folder}/{summary_report_folder}/{chi2_squared_test}.txt', 'a') as file:
            file.write(result) 

    if method_used == fisher_exact_test:
        result += (f'*** {df_name} ***\n\n')
        for row_label, row in df.iterrows():
            for col_label, value in row.items():
                if row_label != col_label:
                    if isinstance(value, (int, float)):
                        if value < report_critical_p_value:
                            result += (f'{row_label:<20} | {col_label:<20} | {round(value, 4):<7} | \n')
        result += ('______________________________\n\n\n')
        with open(f'{result_folder}/{summary_report_folder}/{fisher_exact_test}.txt', 'a') as file:
            file.write(result) 

    elif method_used == yule_Q_test:
        positive_assosication += (f'*** {df_name} ***\n\n Positive Association \n')
        negative_assosication += (f'*** {df_name} ***\n\n Negative Association \n')
        for row_label, row in df.iterrows():
            for col_label, value in row.items():
                if row_label != col_label:
                    if isinstance(value, (int, float)):
                        if round(value, 4) >= report_critical_upper_yule_q:
                            positive_assosication += (f'{row_label:<20} | {col_label:<20} | {round(value, 4):<7} | \n')
                        elif value <= report_critical_lower_yule_q:
                            negative_assosication += (f'{row_label:<20} | {col_label:<20} | {round(value, 4):<7} | \n')
        positive_assosication += ('______________________________\n\n\n')
        negative_assosication += ('______________________________\n\n\n')
        with open(f'{result_folder}/{summary_report_folder}/{yule_Q_test}.txt', 'a') as file:
            file.write(positive_assosication)              
            file.write(negative_assosication)  
    
    elif method_used == proposed_indicator:
        positive_assosication += (f'*** {df_name} ***\n\n Positive Association \n')
        negative_assosication += (f'*** {df_name} ***\n\n Negative Association \n')
        for row_label, row in df.iterrows():
            for col_label, value in row.items():
                if row_label != col_label:
                    if isinstance(value, (int, float)):
                        if value >= report_critical_upper_proposed_indicator:
                            positive_assosication += (f'{row_label:<20} | {col_label:<20} | {round(value, 4):<7} | \n')
                        elif value <= report_critical_lower_proposed_indicator:
                            negative_assosication += (f'{row_label:<20} | {col_label:<20} | {round(value, 4):<7} | \n')
        positive_assosication += ('______________________________\n\n\n')
        negative_assosication += ('______________________________\n\n\n')
        with open(f'{result_folder}/{summary_report_folder}/{proposed_indicator}.txt', 'a') as file:
            file.write(positive_assosication)              
            file.write(negative_assosication)  

  
    elif method_used == conditional_probability:
        positive_assosication += (f'*** {df_name} ***\n\n Positive Association\n')
        for row_label, row in df.iterrows():
            for col_label, value in row.items():
                if row_label != col_label:
                    if isinstance(value, (int, float)):
                        if value >= report_critical_conditional_probability:
                            positive_assosication += (f'{row_label:<20} | {col_label:<20} | {round(value, 4):<7} | \n')
        positive_assosication += ('______________________________\n\n\n')
        with open(f'{result_folder}/{summary_report_folder}/{conditional_probability}.txt', 'a') as file:
            file.write(positive_assosication) 
        
    elif method_used == OR_ratio:
        positive_assosication += (f'*** {df_name} ***\n\n Positive Association \n')
        negative_assosication += (f'*** {df_name} ***\n\n Negative Association \n')
        for row_label, row in df.iterrows():
            for col_label, value in row.items():
                if row_label != col_label:
                    if isinstance(value, (int, float)):
                        if value >= report_critical_OR_value:
                            positive_assosication += (f'{row_label:<20} | {col_label:<20} | {round(value, 4):<7} | \n')
                        elif value <= report_critical_OR_value:
                            negative_assosication += (f'{row_label:<20} | {col_label:<20} | {round(value, 4):<7} | \n')
        positive_assosication += ('______________________________\n\n\n')
        negative_assosication += ('______________________________\n\n\n')
        with open(f'{result_folder}/{summary_report_folder}/{OR_ratio}.txt', 'a') as file:
            file.write(positive_assosication)              
            file.write(negative_assosication)  

    
    elif method_used == pointwise_mutual_info:
        result += (f'*** {df_name} ***\n\n')
        for row_label, row in df.iterrows():
            for col_label, value in row.items():
                if row_label != col_label:
                    if isinstance(value, (int, float)):
                        if value < report_critical_upper_pointwise_mutual_info and value > report_critical_lower_pointwise_mutual_info:
                            result += (f'{row_label:<20} | {col_label:<20} | {round(value, 4):<7} | \n')
        result += ('______________________________\n\n\n')
        with open(f'{result_folder}/{summary_report_folder}/{pointwise_mutual_info} NOTE THIS IS FOR REJECTION.txt', 'a') as file:
            file.write(result) 


    


#________________________________ START _________________________________
# 0. Set Up Your Hyper=parameters
thresholds = [0.3, 0.4, 0.5]  # for binary transfomation  if >= threshold -> V else 0
report_critical_p_value = 0.05  # literature
report_critical_upper_yule_q = 0.5  # literature  moderate positive
report_critical_lower_yule_q = -0.5  # literature moderate negative
report_critical_upper_proposed_indicator = 0.8  # arbitrary
report_critical_lower_proposed_indicator = 0.2  # arbitrary
report_critical_conditional_probability = 0.7  # often considered high
report_critical_OR_value = 1  # > 1 means positive association, negative association otherwise
report_critical_upper_pointwise_mutual_info = 1 # values close to zero?
report_critical_lower_pointwise_mutual_info = -1 # values close to zero?
statistics_methods = [OR_ratio, yule_Q_test, pointwise_mutual_info, proposed_indicator]


# 1. Clean all previous results from the previous run
clean_files_within_directory(result_folder)
clean_files_within_directory(ready_to_transform_folder)
clean_files_within_directory(ready_to_process_folder)

# 2. Import data form the raw_data folder
csv_raw_files = glob.glob(f'{raw_data_folder}/*.csv')

# 3. Prepare data for processing, Part 1
# 3.1. Cleaning and Grouping (from raw_data to ready_to_transform)
for csv_file in csv_raw_files:   
    # Drop unnecessary columns
    #   dropping the first 8 columns that do not have any numerical values, only text
    #   dropping the last 2 columns (differentcon, norel) that do not show in the PDTB-3 sense hierarchy 
    df_leaves = pd.read_csv(csv_file)
    df_leaves = df_leaves.iloc[:, 8:-2]
    file_name = os.path.basename(csv_file)[:-4]  # -4 to remove the '.csv' from the name
    
    # Group to level2
    df_level2 = group_to_level2(df_leaves)

    # Store in a dictionary
    dfs_ready_to_transform[file_name + "_leaves"] = df_leaves  
    dfs_ready_to_transform[file_name + "_level2"] = df_level2

    # Save for checking (saved in ready_to_transform_folder)
    df_leaves.to_csv(f'{ready_to_transform_folder}/{file_name}_leaves.csv', index = False)
    df_level2.to_csv(f'{ready_to_transform_folder}/{file_name}_level2.csv', index = False)
    # dfs_ready_to_process_continuous[file_name + "_leaves"] = df_leaves # we want to consider the data as as without transforming so we can apply pearsom, spearman and kendall before transfomation into binary
   
# 3. Prepare data for processing, Part 2
# 3.2. Data Transformation (from ready_to_transform to ready_to_process)
for df_name, df_ready_to_transform in dfs_ready_to_transform.items():
    # Transform to binary
    for threshold in thresholds:
        # Transform to binary based on the threshold [0.1, 0.2, 0.3, 0.4, 0.5]
        df_transformed_to_binary = transform_to_binary(df_ready_to_transform, threshold)

        # remove all rows and columns that have all zeros
        df_binary_cleaned = remove_all_zeros_columns(df_transformed_to_binary) 

        # Store in a dictionary
        dfs_ready_to_process[f'{df_name}_{threshold}'] = df_binary_cleaned
        
        # Save for checking (saved in ready_to_process_folder)
        df_binary_cleaned.to_csv(f'{ready_to_process_folder}/{binary_folder}/{df_name}_{threshold}_clean.csv')
    
 
global_sheet_col_names = (get_global_Sheet_columns())
global_sheet_df = pd.DataFrame(columns = global_sheet_col_names)

# 4. Processing data
isSkip = False

for df_name, df_ready_to_process in dfs_ready_to_process.items():
    # 4.1. preparing the dictionaries that hold the value matrices (adding the lables)
    column_labels = df_ready_to_process.columns

    chi2_p_value_matrices[df_name] = pd.DataFrame(index = column_labels, columns = column_labels)
    fisher_exact_p_value_matrices[df_name] = pd.DataFrame(index = column_labels, columns = column_labels)
    OR_ratio_matrices[df_name] = pd.DataFrame(index = column_labels, columns = column_labels)
    yuleQ_value_matrices[df_name] = pd.DataFrame(index= column_labels, columns = column_labels)
    pointwise_mutual_info_value_matrices[df_name] = pd.DataFrame(index = column_labels, columns = column_labels)
    conditional_probability_value_matrices[df_name] = pd.DataFrame(index = column_labels, columns = column_labels)
    proposed_indicator_value_matrices[df_name] = pd.DataFrame(index = column_labels, columns = column_labels)



    for i in range(len(df_ready_to_process.columns)):
        for j in range(i+1, len(df_ready_to_process.columns)): # we do from i intead of i+1 to double check our calculaitons by looking at the diagonal (for some methods only)
            isSkip = False

            # 4.2. Extracting pair of sesnes info
            s1_name = df_ready_to_process.columns[i]
            s2_name = df_ready_to_process.columns[j]
            s1_data = df_ready_to_process[s1_name]
            s2_data = df_ready_to_process[s2_name]

            if (i==j):
                isSkip = True # Some of our calculations cannot be done at the diagonals

            # 4.3. Calculating th values for each pair
            contingency_table = get_contingency_matrix(s1_data, s2_data, s1_name, s2_name, df_name, isSkip)
            p_value, method_used, OR_value = get_chi2_or_fisher_p_value_and_OR(contingency_table)   
            yuleQ_value = get_yuleQ_value(contingency_table, isSkip)
            proposed_indicator_value = get_proposed_method_value(contingency_table)
            conditional_probability_value = get_conditional_probability_value(contingency_table)
            pointwise_mutual_info_value = get_pointwise_mutual_info_value(contingency_table)

            global_sheet_df = store_in_global_sheet(global_sheet_df, df_name, s1_name, s2_name, p_value, OR_value, yuleQ_value, pointwise_mutual_info_value, proposed_indicator_value)
            global_sheet_df.to_csv('./global_sheet.csv')

            # 4.4 Storing the pair value in the appropriate cell
            if (method_used == chi2_squared_test):
                chi2_p_value_matrices[df_name].at[s1_name, s2_name] = p_value
                fisher_exact_p_value_matrices[df_name].at[s1_name, s2_name] = method_used
            else:
                chi2_p_value_matrices[df_name].at[s1_name, s2_name] = method_used
                fisher_exact_p_value_matrices[df_name].at[s1_name, s2_name] = p_value
            OR_ratio_matrices[df_name].at[s1_name, s2_name] = OR_value
            yuleQ_value_matrices[df_name].at[s1_name, s2_name] = yuleQ_value
            proposed_indicator_value_matrices[df_name].at[s1_name, s2_name] = proposed_indicator_value
            conditional_probability_value_matrices[df_name].at[s1_name, s2_name] = conditional_probability_value
            pointwise_mutual_info_value_matrices[df_name].at[s1_name, s2_name] = pointwise_mutual_info_value

    # 4.5. Store the result value matrices
    generate_csv(chi2_p_value_matrices[df_name], f'{df_name}_chi2_Pvalue')
    generate_csv(fisher_exact_p_value_matrices[df_name], f'{df_name}_fisherExact_Pvalue')
    generate_csv(OR_ratio_matrices[df_name], f'{df_name}_OR')
    generate_csv(yuleQ_value_matrices[df_name], f'{df_name}_yuleQ')    
    generate_csv(pointwise_mutual_info_value_matrices[df_name], f'{df_name}_pointwiseMutualInfo')
    generate_csv(conditional_probability_value_matrices[df_name], f'{df_name}_conditionalProbability')
    generate_csv(proposed_indicator_value_matrices[df_name], f'{df_name}_proposedIndicator')

    generate_summary_report(df = chi2_p_value_matrices[df_name], df_name= df_name, method_used = chi2_squared_test)
    generate_summary_report(df = fisher_exact_p_value_matrices[df_name], df_name= df_name, method_used = fisher_exact_test)
    generate_summary_report(df = OR_ratio_matrices[df_name], df_name= df_name, method_used = OR_ratio)
    generate_summary_report(df = yuleQ_value_matrices[df_name], df_name= df_name, method_used = yule_Q_test)
    generate_summary_report(df = pointwise_mutual_info_value_matrices[df_name], df_name= df_name, method_used = pointwise_mutual_info)
    generate_summary_report(df = conditional_probability_value_matrices[df_name], df_name= df_name, method_used = conditional_probability)
    generate_summary_report(df = proposed_indicator_value_matrices[df_name], df_name= df_name, method_used = proposed_indicator)


