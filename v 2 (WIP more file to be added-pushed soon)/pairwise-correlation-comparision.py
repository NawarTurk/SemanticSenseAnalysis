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
analysis_value_matrices_folder = 'analysis_value_matrices'
heatmap_folder = 'heatmaps'
csv_files_folder = 'csv_files'
contingency_table_folder = 'contingency_tables'
# binary_corr_matrix_folder = 'binary corr matrix'
# # continuous_corr_matrix_folder = 'continuous corr matrix'
# binary_heatmap_folder = 'binary heatmap'
# continuous_heatmap_folder = 'continuous heatmap'
# report_file = f'{result_folder}/report.txt'



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
        
def get_contingency_matrix(s1, s2, df_name, isSkip):
    contingency_table = pd.crosstab(s1, s2)
    contingency_table_str = str(contingency_table)

    if (not isSkip):
        with open(f'./{result_folder}/{binary_folder}/{contingency_table_folder}/{df_name}.txt', 'a') as file:
            file.write(contingency_table_str + '\n\n' + '-.-.-.-.-.-.-.-.-.-')
    return contingency_table

def get_chi2_p_value(contingency_table):
    stat, p, dof, expected = chi2_contingency(contingency_table, correction= False)
    # correction= True means Yates's correction
    # The effect of Yates's correction is to prevent overestimation of statistical significance for small data. 
    # This formula is chiefly used when at least one cell of the table has an expected count smaller than 5
    return p

def get_yuleQ_value(contingency_table, isSkip):
    one_one = contingency_table.loc[1,1]
    one_zero = contingency_table.loc[1,0]
    zero_one = contingency_table.loc[0,1]
    zero_zero = contingency_table.loc[0,0]
    if (isSkip):
        return 'X'
    OR = one_one*zero_zero/(one_zero*zero_one)
    yule_Q = (OR-1)/(OR+1)
    return yule_Q

def get_proposed_method_value(contingency_table):
    one_one = contingency_table.loc[1,1]
    one_zero = contingency_table.loc[1,0]
    zero_one = contingency_table.loc[0,1]
    proposed_method_value = (one_one/(one_one + one_zero + zero_one))
    return proposed_method_value

def generate_csv(matrix_value_df, title):
    matrix_value_df.to_csv(f'./{result_folder}/{binary_folder}/{analysis_value_matrices_folder}/{csv_files_folder}/{title}.csv')


 
contingency_tables = {}
chi2_p_value_matrices = {}
yuleQ_value_matrices = {}
proposed_indicator_value_matrices = {}
isSkip = False


for df_name, df_ready_to_process in dfs_ready_to_process_binary.items():

    column_labels = df_ready_to_process.columns
    contingency_tables[df_name] = pd.DataFrame(index=column_labels, columns=column_labels)
    chi2_p_value_matrices[df_name] = pd.DataFrame(index=column_labels, columns= column_labels)
    yuleQ_value_matrices[df_name] = pd.DataFrame(index= column_labels, columns= column_labels)
    proposed_indicator_value_matrices[df_name] = pd.DataFrame(index= column_labels, columns= column_labels)
        

    for i in range(len(df_ready_to_process.columns)):
        for j in range(i, len(df_ready_to_process.columns)): # we do from i intead of i+1 to double check our calculaitons by looking at the diagonal
            s1_name = df_ready_to_process.columns[i]
            s2_name = df_ready_to_process.columns[j]
            s1_data = df_ready_to_process[s1_name]
            s2_data = df_ready_to_process[s2_name]

            if (i==j):
                isSkip = True

            contingency_table = get_contingency_matrix(s1_data, s2_data, df_name, isSkip)
            # check point
            # print(contingency_table) 

            chi2_p_value = get_chi2_p_value(contingency_table)
            # check point
            # print(f'chi2_p_value = {chi2_p_value}')
            
            yuleQ_value = get_yuleQ_value(contingency_table, isSkip)
            # check point
            # print(f'yuleQ_value = {yuleQ_value}')
            
            proposed_indicator_value = get_proposed_method_value(contingency_table)
            # check point
            # print(f'proposed_method_value = {get_proposed_method_value(contingency_table)*100}%')

            # creating a matrix for the results, one for each a dataframe(here we have only df of discogem at leaves level transfered to binar at 0.3)
            contingency_tables[df_name].at[s1_name, s2_name] = contingency_table
            chi2_p_value_matrices[df_name].at[s1_name, s2_name] = chi2_p_value
            yuleQ_value_matrices[df_name].at[s1_name, s2_name] = yuleQ_value
            proposed_indicator_value_matrices[df_name].at[s1_name, s2_name] = proposed_indicator_value

            isSkip = False

    generate_csv(chi2_p_value_matrices[df_name], f'{df_name}_chi2_P_value')
    generate_csv(yuleQ_value_matrices[df_name], f'{df_name}_yuleQ_value')
    generate_csv(proposed_indicator_value_matrices[df_name], f'{df_name}_proposed_indicator_value')


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





    




