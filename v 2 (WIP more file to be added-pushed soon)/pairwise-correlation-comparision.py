import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import os
from scipy.stats import chi2_contingency


# hyper=parameters
# UPPER_CORR_THRESHOLD = 0.7  # recommended threshold for + correlation 
# LOWER_CORR_THRESHOLD = -0.7 # recommended threshold for - correlation 
thresholds = [0.3]  # for binary transfomation  if > threshold -> 1 else 0
correlation_coefficients = ['pearson', 'spearman', 'kendall']
# sense_of_interest = ['synchronous',	'precedence',	'reason',	'result',	'arg1-as-denier',	'arg2-as-denier',	'contrast',	'similarity',	'conjunction',	'arg2-as-instance',	'arg1-as-detail',	'arg2-as-detail', 'instantiation',	'level-of-detail', 'asynchronous',	'cause',	'concession']


# folder names
raw_data_folder = '0_raw_data'
ready_to_transform_folder = '1_ready_to_transform'
ready_to_process_folder = '2_ready_to_process'
result_folder = '3_results'
binary_folder = 'binary'
continuous_folder = 'continuous'
binary_chi2Pvalue_matrices_folder = 'binary_chi2Pvalue_matrices'
binary_corr_matrix_folder = 'binary corr matrix'
continuous_corr_matrix_folder = 'continuous corr matrix'
binary_heatmap_folder = 'binary heatmap'
continuous_heatmap_folder = 'continuous heatmap'
report_file = f'{result_folder}/report.txt'



# dataframe dictionaries
dfs_ready_to_transform = {}
dfs_ready_to_process_binary = {}
dfs_ready_to_process_continuous = {}



# Clean all the files of the previous run to generate new results
def clean_files_within_directory(directory_name):
    # Construct the full path to the directory relative to the current script
    directory_path = os.path.join(os.path.dirname(__file__), directory_name)  
    # Walk through the directory
    for root, dirs, files in os.walk(directory_path):
        # Remove each file in the root and subdirectories
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)

# def annotate_sense_of_interest(df, list_of_sense):
#     for col in df.columns:
#         if col in list_of_sense:
#             df.rename(columns = {col: col + '+++'}, inplace = True)

# def convert_to_level2(df):
#     df_level2 = pd.DataFrame()
#     df_level2['synchronous'] = df['synchronous']
#     df_level2['asynchronous'] = df['precedence'] + df['succession'] 
#     df_level2['cause'] = df['reason'] + df['result'] 
#     df_level2['condition'] = df['arg1-as-cond'] + df['arg2-as-cond'] 
#     df_level2['negative-condition'] = df['arg1-as-negcond'] + df['arg2-as-negcond'] 
#     df_level2['purpose'] = df['arg1-as-goal'] + df['arg2-as-goal'] 
#     df_level2['concession'] = df['arg1-as-denier'] + df['arg2-as-denier'] 
#     df_level2['contrast'] = df['contrast']
#     df_level2['similarity'] = df['similarity']
#     df_level2['conjunction'] = df['conjunction']
#     df_level2['disjunction'] = df['disjunction']
#     df_level2['instantiation'] = df['arg1-as-instance'] + df['arg2-as-instance'] 
#     df_level2['level-of-detail'] = df['arg1-as-detail'] + df['arg2-as-detail'] 
#     df_level2['equivalence'] = df['equivalence']
#     df_level2['manner'] = df['arg1-as-manner'] + df['arg2-as-manner'] 
#     df_level2['exception'] = df['arg1-as-excpt'] + df['arg2-as-excpt'] 

    # not both files have the 'arg1-as-subst' column in it 
    # if 'arg1-as-subst' in df.columns: 
    #     df_level2['substitution'] = df['arg1-as-subst'] + df['arg2-as-subst'] 
    # else:
    #     df_level2['substitution'] = df['arg2-as-subst'] 
    # return df_level2


# def convert_to_level1(df_level2):
#     df_level1 = pd.DataFrame()
#     df_level1['temporal'] = df_level2['synchronous'] + df_level2['asynchronous'] 
#     df_level1['contingency'] = df_level2['cause'] + df_level2['condition'] + df_level2['negative-condition'] + df_level2['purpose'] 
#     df_level1['comparision'] = df_level2['concession'] + df_level2['contrast'] + df_level2['similarity']
#     df_level1['expansion'] = df_level2['conjunction'] + df_level2['disjunction'] + df_level2['equivalence'] + df_level2['exception'] + \
#                              df_level2['instantiation'] + df_level2['level-of-detail'] + df_level2['manner'] + df_level2['substitution']
#     return df_level1


# Transform to binary data
def transform_to_binary(df_ready_to_transform, threshold):
    return df_ready_to_transform.map(lambda x: 1 if x >= threshold else 0)

# Transfomr to continuous Space
# def transform_to_CLR(df_ready_to_transform):
#     # Replace zeros with a small positive value to avoid division by zero or log of zero
#     df_no_zeros = df_ready_to_transform.replace(0, 1e-5).values  # Convert to numpy array for efficiency
    
#     # Calculate the geometric mean of each row
#     geometric_mean = np.exp(np.mean(np.log(df_no_zeros), axis=1)).reshape(-1, 1)  # Reshape for broadcasting
#     # axis=1 means the operation is performed row-wise across columns (across the rows).
    
#     # Apply the CLR transformation: log(x_i / geometric_mean(x))
#     df_clr = np.log(df_no_zeros / geometric_mean)
    
#     # Convert back to a pandas DataFrame, if necessary
#     df_clr = pd.DataFrame(df_clr, index=df_ready_to_transform.index, columns=df_ready_to_transform.columns)
    
#     return df_clr


# def transform_to_ALR(df_ready_to_transform):
#     pass
    
    
# def transform_to_ILR(df_ready_to_transform):
#     pass


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


# def plot_scatter_matrix(df, df_name, set_limit_1):
#     num_columns = len(df.columns)
#     fig, ax = plt.subplots(nrows=num_columns, ncols=num_columns, figsize=(num_columns*4, num_columns*4))
    
#     # Loop over all the columns
#     for i in range(num_columns):
#         for j in range(i + 1, num_columns):
#             # Scatter plot for each pair of columns
#             ax[i, j].scatter(df.iloc[:, j], df.iloc[:, i]) 
#             if set_limit_1:
#                 ax[i, j].set_xlim(0, 1)  # Set x-axis to extend to 1
#                 ax[i, j].set_ylim(0, 1)  # Set y-axis to extend to 1
#             ax[i, j].set_xlabel(df.columns[j])
#             ax[i, j].set_ylabel(df.columns[i])
                
#     plt.tight_layout()
#     plt.savefig(f'{result_folder}/{df_name}.png')

# def plot_histogram_matrix(df, df_name):
#     cols = df.columns
#     n_cols = 3
#     n_rows = np.ceil(len(cols) / n_cols)
#     plt.figure(figsize(15, 4*))



#______________________________________________ START _________________________________

# CLEAN ALL PREVIOUS RESULTS FROM THE PREVIOUS RUN
clean_files_within_directory(result_folder)
clean_files_within_directory(ready_to_transform_folder)
clean_files_within_directory(ready_to_process_folder)


# PREPARING THE DATA FOR PROCESSING LATER ON

# *** Import all files in rawa_data folder
csv_raw_files = glob.glob(f'{raw_data_folder}/*.csv')

# *** Data Preparation   (raw_data --> read_to_transform)
# prepare the dataframes (dropping out unecessary columns):
#   dropping the first 8 columns that do not have any numerical values, only text
#   dropping the last 2 columns (differentcon, norel) that do not show in the PDTB-3 sense hierarchy 
for csv_file in csv_raw_files:
    # for every file in the raw_data folder do below (currenly we only doing discogem)
    df_leaves = pd.read_csv(csv_file)
    df_leaves = df_leaves.iloc[:, 8:-2]  # dropping out unecessary columns  check the last two columns  +++++ ATTENTION +++++
    file_name = os.path.basename(csv_file)[:-4]  # -4 to remove the '.csv' from the name

    # df_level2 = convert_to_level2(df_leaves)
    # df_level1 = convert_to_level1(df_level2)

    # annotate_sense_of_interest(df_leaves, sense_of_interest)
    # annotate_sense_of_interest(df_level2, sense_of_interest)

    # dfs_ready_to_transform is a dicitoanry that has all the dataframes that we will transform
    #   currently we are processing discogem at leaves level, later on we can add discogem level2, qadc leaves level and qadc level2
    dfs_ready_to_transform[file_name + "_leaves"] = df_leaves 
    # dfs_ready_to_transform[file_name + "_level2"] = df_level2
    # dfs_ready_to_transform[file_name + "_level1"] = df_level1

    # saving the file in the ready_to_transform folder so we can check everything is ok
    df_leaves.to_csv(f'{ready_to_transform_folder}/{file_name}_leaves.csv', index = False)
    # df_level2.to_csv(f'{ready_to_transform_folder}/{file_name}_level2.csv', index = False)
    # df_level1.to_csv(f'{ready_to_transform_folder}/{file_name}_level1.csv', index = False)

    # plot_scatter_matrix(df_leaves, f'{file_name}_leaves', True)
    # plot_scatter_matrix(df_level2, f'{file_name}_level2', True)
    # plot_scatter_matrix(df_level1, f'{file_name}_level1', True)

    # adding the raw data for analysis with correlation coefficients without any transformation
    # explain it
    dfs_ready_to_process_continuous[file_name + "_leaves"] = df_leaves # we want to consider the data as as without transforming so we can apply pearsom, spearman and kendall before transfomation into binary
   


# *** Data Transformation (read_to_transform --> ready to process)
# we now have one dataframe that is (discogem leaves level, but later we can have discogem level2, qadc leaves level and qadc level2)
for df_name, df_ready_to_transform in dfs_ready_to_transform.items():
    # Transform to binary ( we have thresholds [0.1, 0.2, 0.3, 0.4])
    for threshold in thresholds:
        # transform to binary based on the threshold
        df_transformed_to_binary = transform_to_binary(df_ready_to_transform, threshold)

        # dfs_ready_to_process_binary is a dictionary that has all the data frames transformed into binary
        # now we only have discogem at leaves level at threshold 0.1, 0.2, 0.3, 0.4 
        # later we can do the same for discogem level2, and do the same for qadc
        dfs_ready_to_process_binary[df_name + f'_binary_{threshold}'] = df_transformed_to_binary
        #save it so can we examine it
        df_transformed_to_binary.to_csv(f'{ready_to_process_folder}/{binary_folder}/{df_name}_binary_{threshold}.csv')
    
    # Transfrom to continuous
        # transform
    # df_transformed_to_CLR = transform_to_CLR(df_ready_to_transform)
    # # df_transformed_to_ALR = transform_to_ALR(df_ready_to_transform)
    # # df_transformed_to_ILR = transform_to_ILR(df_ready_to_transform)
    #     # store
    # dfs_ready_to_process_continuous[df_name + '_continuous_CLR'] = df_transformed_to_CLR
    # # dfs_ready_to_process_continuous[df_name + '_continuous_ALR'] = df_transformed_to_ALR
    # # dfs_ready_to_process_continuous[df_name + '_continuous_ILR'] = df_transformed_to_ILR
    #     # save
    # df_transformed_to_CLR.to_csv(f'{ready_to_process_folder}/{continuous_folder}/{df_name}_CLR.csv')
    # # df_transformed_to_ALR.to_csv(f'{ready_to_process_folder}/{continuous_folder}/{df_name}_ALR.csv')
    # # df_transformed_to_ILR.to_csv(f'{ready_to_process_folder}/{continuous_folder}/{df_name}_ILR.csv')

    # plot_scatter_matrix(df_transformed_to_CLR, f'CLR_{df_name}', False)
# THE DATA IS NOW READY TO BE PROCESSED
        
def get_contingency_matrix(s1, s2):
    contingency_table = pd.crosstab(s1, s2)
    return contingency_table


# def phi_coefficient(col1, col2):
#     contingency_table = pd.crosstab(col1, col2)
#     print('contingency_tabl')
#     print(contingency_table)
#     chi2, _, _, _ = chi2_contingency(contingency_table)
#     print("chi2 here")
#     print(chi2)
#     n = contingency_table.values.sum()
#     print("Total observations (n):", n)
#     print("n here")
#     print(n)
#     phi = np.sqrt(chi2 / n)
#     print("here")
#     print(phi)
#     return phi

# def phi_coefficient_matrix(df):
#     cols = df.columns
#     n_cols = len(cols)
#     phi_matrix = pd.DataFrame(np.zeros((n_cols, n_cols)), index = cols, columns = cols)
    
#     for i in range(n_cols):
#         for j in range(i, n_cols):
#             phi_value = phi_coefficient(df.iloc[:, i], df.iloc[:, j])
#             print(phi_value)
#             phi_matrix.loc[cols[i], cols[j]] =  phi_value
#             phi_matrix.loc[cols[j], cols[i]] = phi_value
#     return phi_matrix


#_Data Processing_  
for df_name, df_ready_to_process in dfs_ready_to_process_binary.items():
    for i in range(len(df_ready_to_process.columns)):
        for j in range(i+1, len(df_ready_to_process.columns)):


            s1_name = df_ready_to_process.columns[i]
            s2_name = df_ready_to_process.columns[j]

            s1_data = df_ready_to_process[s1_name]
            s2_data = df_ready_to_process[s2_name]

            contingency_table = get_contingency_matrix(s1_data, s2_data)
            print(contingency_table)
            print(type(contingency_table))




            # contingency_table = get_contingency_matrix()


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





    




