import pandas as pd
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import numpy as np
import glob
import os
from scipy.stats import chi2_contingency
from scipy.stats import fisher_exact


# Hyper=parameters
# UPPER_CORR_THRESHOLD = 0.7  # recommended threshold for + correlation 
# LOWER_CORR_THRESHOLD = -0.7 # recommended threshold for - correlation 
#   binary converstion threshold
thresholds = [0.3, 0.4, 0.5]  # for binary transfomation  if > threshold -> 1 else 0
correlation_coefficients = ['pearson', 'spearman', 'kendall']  # for continuous data (no binary transformation)
report_critical_p_value = 0.05  # literature
report_critical_upper_yule_q = 0.8  # literature
report_critical_lower_yule_q = -0.8  # literature
report_critical_upper_proposed_indicator = 0.8  # arbitrary
report_critical_lower_proposed_indicator = 0.2  # arbitrary

# Dataframe dictionaries
dfs_ready_to_transform = {}
dfs_ready_to_process_binary = {}
dfs_ready_to_process_continuous = {}

# Value matrices dictionaries     
contingency_tables = {}
chi2_p_value_matrices = {}
fisher_exact_p_value_matrices = {}
yuleQ_value_matrices = {}
proposed_indicator_value_matrices = {}

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
report_folder = 'report'
summary_report_folder = 'summary_report'

# Others
not_applicable = 'NA'
fisher_exact_test ='fisher'
chi2_squared_test = 'chi2'
yule_Q_test = 'yuleQ'
proposed_indicator = 'proposed_indicator'



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
    return df_ready_to_transform.map(lambda x: 1 if x >= threshold else 0)

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
    s1 = pd.Categorical(s1, categories=[0, 1]) # we had to explictly mention the categories because it will not be added to the table if they a col or a row has values of zeros
    s2 = pd.Categorical(s2, categories=[0, 1])
    contingency_table = pd.crosstab(s1, s2, rownames=[s1_name], colnames=[s2_name], dropna=False)
    contingency_table_str = str(contingency_table)

    if (not isSkip):
        with open(f'./{result_folder}/{binary_folder}/{contingency_table_folder}/{df_name}.txt', 'a') as file:
            file.write(contingency_table_str + '\n\n' + '-.-.-.-.-.-.-.-.-.-\n')
    
    return contingency_table

def get_chi2_or_fisher_p_value(contingency_table):
    """
    Calculates the p-value using the Chi-squared test or Fisher's Exact test based on the suitability for the provided contingency table.

    This function first attempts to compute the p-value using the Chi-squared test. 
    If the Chi-squared test is not applicable due to the expected table (derived from the contingency table) contains a zero cell, 
    the function falls back to Fisher's Exact test.

    Parameters:
    - contingency_table (pd.DataFrame): A 2x2 contingency table.

    Returns:
    - tuple: A tuple containing the p-value and a string indicating which test was used 
    """
    try:
        stat, chi2_p, dof, expected = chi2_contingency(contingency_table, correction= False)
    except ValueError as e:
        _, fisher_exact_p = scipy.stats.fisher_exact(contingency_table)
        return (fisher_exact_p, fisher_exact_test)
    return (chi2_p, chi2_squared_test)

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

    one_one = contingency_table.loc[1,1]
    one_zero = contingency_table.loc[1,0]
    zero_one = contingency_table.loc[0,1]
    zero_zero = contingency_table.loc[0,0]
    if (isSkip):
        # we set isSkip to true when we are comparing the same sense (i=j)
        return 'NA'
    if (one_one*one_zero*zero_one*zero_zero == 0):  
        # none of the contingency table cells can be zero
        return 'NA'
    numerator = (one_one * zero_zero) - (one_zero * zero_one)
    denominator = (one_one * zero_zero) + (one_zero * zero_one)
    yules_q = numerator / denominator

    return yules_q

def get_proposed_method_value(contingency_table):
    one_one = contingency_table.loc[1,1]
    one_zero = contingency_table.loc[1,0]
    zero_one = contingency_table.loc[0,1]

    numerator = one_one
    denominator = one_one + one_zero + zero_one

    if denominator == 0:
        return not_applicable  

    proposed_method_value = numerator/denominator
    return proposed_method_value



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

def generate_summary_report(df, df_name, method_used):
    result = ''
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

#________________________________ START _________________________________

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


    dfs_ready_to_process_continuous[file_name + "_leaves"] = df_leaves # we want to consider the data as as without transforming so we can apply pearsom, spearman and kendall before transfomation into binary
   
# 3. Prepare data for processing, Part 2
# 3.2. Data Transformation (from ready_to_transform to ready_to_process)

for df_name, df_ready_to_transform in dfs_ready_to_transform.items():
    # Transform to binary
    for threshold in thresholds:
        # Transform to binary based on the threshold [0.1, 0.2, 0.3, 0.4, 0.5]
        df_transformed_to_binary = transform_to_binary(df_ready_to_transform, threshold)

        # Store in a dictionary
        dfs_ready_to_process_binary[df_name + f'_binary_thresholdOf{threshold}'] = df_transformed_to_binary
        
        # Save for checking (saved in ready_to_process_folder)
        df_transformed_to_binary.to_csv(f'{ready_to_process_folder}/{binary_folder}/{df_name}_binary_{threshold}.csv')
    
 
# 4. Processing data
isSkip = False

for df_name, df_ready_to_process in dfs_ready_to_process_binary.items():
    # 4.1. preparing the dictionaries that hold the value matrices (adding the lables)
    column_labels = df_ready_to_process.columns
    contingency_tables[df_name] = pd.DataFrame(index = column_labels, columns =column_labels)
    chi2_p_value_matrices[df_name] = pd.DataFrame(index = column_labels, columns = column_labels)
    fisher_exact_p_value_matrices[df_name] = pd.DataFrame(index = column_labels, columns = column_labels)
    yuleQ_value_matrices[df_name] = pd.DataFrame(index= column_labels, columns = column_labels)
    proposed_indicator_value_matrices[df_name] = pd.DataFrame(index = column_labels, columns = column_labels)
        

    for i in range(len(df_ready_to_process.columns)):
        for j in range(i, len(df_ready_to_process.columns)): # we do from i intead of i+1 to double check our calculaitons by looking at the diagonal (for some methods only)
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
            p_value, method_used = get_chi2_or_fisher_p_value(contingency_table)   
            yuleQ_value = get_yuleQ_value(contingency_table, isSkip)
            proposed_indicator_value = get_proposed_method_value(contingency_table)

            # 4.4 Storing the pair value in the appropriate cell
            contingency_tables[df_name].at[s1_name, s2_name] = contingency_table

            if (method_used == chi2_squared_test):
                chi2_p_value_matrices[df_name].at[s1_name, s2_name] = p_value
                fisher_exact_p_value_matrices[df_name].at[s1_name, s2_name] = method_used

            else:
                chi2_p_value_matrices[df_name].at[s1_name, s2_name] = method_used
                fisher_exact_p_value_matrices[df_name].at[s1_name, s2_name] = p_value

            yuleQ_value_matrices[df_name].at[s1_name, s2_name] = yuleQ_value
            proposed_indicator_value_matrices[df_name].at[s1_name, s2_name] = proposed_indicator_value

    # 4.5. Store the result value matrices
    generate_csv(chi2_p_value_matrices[df_name], f'{df_name}_chi2_P_value')
    generate_csv(fisher_exact_p_value_matrices[df_name], f'{df_name}_fisher_exact_P_value')
    generate_csv(yuleQ_value_matrices[df_name], f'{df_name}_yuleQ_value')
    generate_csv(proposed_indicator_value_matrices[df_name], f'{df_name}_proposed_indicator_value')

    generate_summary_report(df=chi2_p_value_matrices[df_name], df_name= df_name, method_used = chi2_squared_test)
    generate_summary_report(yuleQ_value_matrices[df_name], df_name= df_name, method_used = yule_Q_test)
    generate_summary_report(chi2_p_value_matrices[df_name], df_name= df_name, method_used = proposed_indicator)





# def save_to_report(data_name, first_sense, seconde_sense, corr_value):
#     with open(report_file, 'a+') as file:
#         file.write(f'{first_sense:<17}|{seconde_sense:<17}|{corr_value:<+7.2f}|  {data_name}\n')



# def create_and_save_heatmap(corr_matrix, file_path, heatmap_title):
#     plt.figure(figsize=(8, 6))
#     ax = sns.heatmap(corr_matrix, annot=False, fmt=".2f", cmap='coolwarm', 
#                 cbar=True, vmin=-1, vmax=1, linewidths=1, linecolor='black')

#     # finding the max and min value
#     numpy_corr_matrix = corr_matrix.to_numpy()  # need to be transformed to a numpy matrix
#     np.fill_diagonal(numpy_corr_matrix, np.nan)  # because they are all 1s, removing them to reduce noise
#     flattened_corr_matrix = numpy_corr_matrix.flatten()  # to be able to calculate the min and max
#     max_corr_value = np.nanmax(flattened_corr_matrix)
#     min_corr_value = np.nanmin(flattened_corr_matrix)

#     # only annotating cells that are beyond the corr_threshold, and the max/min values
#     for i in range(corr_matrix.shape[0]):
#         for j in range(i+1, corr_matrix.shape[1]):
#             corr_value = corr_matrix.iloc[i, j]
#             if corr_value >= UPPER_CORR_THRESHOLD or corr_value == max_corr_value or corr_value <= LOWER_CORR_THRESHOLD or corr_value == min_corr_value:
#                 text = ax.text(j + 0.5, i + 0.5, f'{corr_value:.2f}',
#                             ha="center", va="center", color="black")
#                 first_sense = corr_matrix.columns[i]  # a sense in the correlation
#                 seconde_sense = corr_matrix.columns[j]  # a sense in the correlation
#                 save_to_report(heatmap_title, first_sense, seconde_sense, corr_value)
                       
#     # Customize the color bar
#     cbar = ax.collections[0].colorbar
#     cbar.set_ticks(np.arange(-1, 1.1, 0.1))  # Setting ticks from -1 to 1 at intervals of 0.1
#     cbar.set_ticklabels([f'{tick:.1f}' for tick in np.arange(-1, 1.1, 0.1)])  # Formatting tick labels

#     plt.title(f'{heatmap_title}_heatmap')
#     plt.tight_layout()
#     plt.savefig(f'{file_path}/{heatmap_title}_heatmap.png')
#     plt.close()

    # corr_matrix = phi_coefficient_matrix(df_ready_to_process)
    # corr_matrix.to_csv(f'{result_folder}/{binary_folder}/{binary_corr_matrix_folder}/{df_name}_Phi_corr_matrix.csv')

    # heatmap_file_path = f'{result_folder}/{binary_folder}/{binary_heatmap_folder}'
    # heatmap_title = df_name + '_Phi_corr'
    # create_and_save_heatmap(corr_matrix, heatmap_file_path, heatmap_title)

# for df_name, df_ready_to_process in dfs_ready_to_process_continuous.items():
#     for corr_coefficient in correlation_coefficients:

#         corr_matrix = df_ready_to_process.corr(corr_coefficient)

#         corr_matrix.to_csv(f'{result_folder}/{continuous_folder}/{continuous_corr_matrix_folder}/{df_name}_{corr_coefficient}_corr_matrix.csv')

#         heatmap_file_path = f'{result_folder}/{continuous_folder}/{continuous_heatmap_folder}'
#         heatmap_title = df_name + '_' + corr_coefficient
#         create_and_save_heatmap(corr_matrix, heatmap_file_path, heatmap_title)





    




#
    
# def convert_to_level1(df_level2):
#     df_level1 = pd.DataFrame()
#     df_level1['temporal'] = df_level2['synchronous'] + df_level2['asynchronous'] 
#     df_level1['contingency'] = df_level2['cause'] + df_level2['condition'] + df_level2['negative-condition'] + df_level2['purpose'] 
#     df_level1['comparision'] = df_level2['concession'] + df_level2['contrast'] + df_level2['similarity']
#     df_level1['expansion'] = df_level2['conjunction'] + df_level2['disjunction'] + df_level2['equivalence'] + df_level2['exception'] + \
#                              df_level2['instantiation'] + df_level2['level-of-detail'] + df_level2['manner'] + df_level2['substitution']
#     return df_level1
    

    # def annotate_sense_of_interest(df, list_of_sense):
#     for col in df.columns:
#         if col in list_of_sense:
#             df.rename(columns = {col: col + '+++'}, inplace = True)
    


    # sense_of_interest = ['synchronous',	'precedence',	'reason',	'result',	'arg1-as-denier',	'arg2-as-denier',	'contrast',	'similarity',	'conjunction',	'arg2-as-instance',	'arg1-as-detail',	'arg2-as-detail', 'instantiation',	'level-of-detail', 'asynchronous',	'cause',	'concession']
