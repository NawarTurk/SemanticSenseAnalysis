import pandas as pd
import numpy as np
import glob
import os
from scipy.stats import chi2_contingency
from scipy.stats import fisher_exact
import yaml

from statistical_analysis import (
    get_chi2_or_fisher_p_value_and_OR,
    get_yuleQ_value,
    get_proposed_method_value,
    get_conditional_probability_value,
    get_pointwise_mutual_info_value
)

from file_management import (
    create_required_directories,
    clean_files_within_directory,
    generate_csv,
    save_expected_table,
    generate_summary_report
)

from data_processing import (
    transform_to_binary,
    remove_all_zeros_columns,
    get_contingency_matrix,
    group_to_level2
)

# Dataframe dictionaries
dfs_ready_to_transform = {}
dfs_ready_to_process = {}

# Value matrices dictionaries     
chi2_p_value_matrices = {}
fisher_exact_p_value_matrices = {}
OR_ratio_matrices = {}
yuleQ_value_matrices = {}
proposed_indicator_value_matrices = {}
conditional_probability_value_matrices = {}
pointwise_mutual_info_value_matrices = {}

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


def validate_config_keys(config, required_keys):
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required key: {key}")

required_keys = ['paths', 'constants', 'hyperparameters']
validate_config_keys(config, required_keys)

# Load paths from YAML config
raw_data_folder = config['paths']['raw_data_folder']
ready_to_transform_folder = config['paths']['ready_to_transform_folder']
ready_to_process_folder = config['paths']['ready_to_process_folder']
result_folder = config['paths']['result_folder']
binary_folder = config['paths']['binary_folder']
csv_files_folder = config['paths']['csv_files_folder']
contingency_table_folder = config['paths']['contingency_table_folder']
expected_table_folder = config['paths']['expected_table_folder']
summary_report_folder = config['paths']['summary_report_folder']

# Load constants from YAML config
not_applicable = config['constants']['not_applicable']
fisher_exact_test = config['constants']['fisher_exact_test']
chi2_squared_test = config['constants']['chi2_squared_test']
yule_Q_test = config['constants']['yule_Q_test']
proposed_indicator = config['constants']['proposed_indicator']
conditional_probability = config['constants']['conditional_probability']
OR_ratio = config['constants']['OR_ratio']
pointwise_mutual_info = config['constants']['pointwise_mutual_info']

# Load hyperparameters from YAML config
thresholds = config['hyperparameters']['thresholds']
report_critical_p_value = config['hyperparameters']['report_critical_p_value']
report_critical_upper_yule_q = config['hyperparameters']['report_critical_upper_yule_q']
report_critical_lower_yule_q = config['hyperparameters']['report_critical_lower_yule_q']
report_critical_upper_proposed_indicator = config['hyperparameters']['report_critical_upper_proposed_indicator']
report_critical_lower_proposed_indicator = config['hyperparameters']['report_critical_lower_proposed_indicator']
report_critical_conditional_probability = config['hyperparameters']['report_critical_conditional_probability']
report_critical_OR_value = config['hyperparameters']['report_critical_OR_value']
report_critical_upper_pointwise_mutual_info = config['hyperparameters']['report_critical_upper_pointwise_mutual_info']
report_critical_lower_pointwise_mutual_info = config['hyperparameters']['report_critical_lower_pointwise_mutual_info']

#________________________________ START _________________________________

create_required_directories()

# 1. Clean all previous results from the previous run
clean_files_within_directory(result_folder)
clean_files_within_directory(ready_to_transform_folder)
clean_files_within_directory(ready_to_process_folder)

# 2. Import data form the raw_data folder
csv_raw_files = glob.glob(os.path.join(raw_data_folder, '*.csv'))

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
    df_leaves.to_csv(os.path.join(ready_to_transform_folder, f'{file_name}_leaves.csv'), index=False)
    df_level2.to_csv(os.path.join(ready_to_transform_folder, f'{file_name}_level2.csv'), index=False)

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
        df_binary_cleaned.to_csv(os.path.join(ready_to_process_folder, binary_folder, f'{df_name}_{threshold}_clean.csv'))

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
            p_value, method_used, OR_value = get_chi2_or_fisher_p_value_and_OR(contingency_table, df_name)   
            yuleQ_value = get_yuleQ_value(contingency_table, isSkip)
            proposed_indicator_value = get_proposed_method_value(contingency_table)
            conditional_probability_value = get_conditional_probability_value(contingency_table)
            pointwise_mutual_info_value = get_pointwise_mutual_info_value(contingency_table)

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


