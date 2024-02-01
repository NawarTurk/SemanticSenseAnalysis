import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import os

# hyper=parameters
UPPER_CORR_THRESHOLD = 0.7  # recommended threshold for + correlation 
LOWER_CORR_THRESHOLD = -0.7 # recommended threshold for - correlation 
exist_thresholds = [0.1, 0.2, 0.3, 0.4]  # for binary transfomation  if > threshold -> 1 else 0

# folder names
raw_data_folder = 'raw_data'
ready_to_transform_folder = 'ready_to_transform'
ready_to_process_folder = 'ready_to_process'
binary_folder = 'binary'

# dataframe dictionaries
dfs_ready_to_transform = {}
dfs_ready_to_process = {}



def convert_to_level2(df):
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

    # not both files have the 'arg1-as-subst' column in it 
    if 'arg1-as-subst' in df.columns: 
        df_level2['substitution'] = df['arg1-as-subst'] + df['arg2-as-subst'] 
    else:
        df_level2['substitution'] = df['arg2-as-subst'] 
    return df_level2


def convert_to_level1(df_level2):
    df_level1 = pd.DataFrame()
    df_level1['temporal'] = df_level2['synchronous'] + df_level2['asynchronous'] 
    df_level1['contingency'] = df_level2['cause'] + df_level2['condition'] + df_level2['negative-condition'] + df_level2['purpose'] 
    df_level1['comparision'] = df_level2['concession'] + df_level2['contrast'] + df_level2['similarity']
    df_level1['expansion'] = df_level2['conjunction'] + df_level2['disjunction'] + df_level2['equivalence'] + df_level2['exception'] + \
                             df_level2['instantiation'] + df_level2['level-of-detail'] + df_level2['manner'] + df_level2['substitution']
    return df_level1

def transform_to_binary(df_ready_to_transform, threshold):
    return df_ready_to_transform.map(lambda x: 1 if x > threshold else 0)



csv_raw_files = glob.glob(f'{raw_data_folder}/*.csv')

# prepare the dataframes at the three levels for all the raw data files
for csv_file in csv_raw_files:
    df_leaves = pd.read_csv(csv_file)
    df_leaves = df_leaves.iloc[:, 8:-2]  # dropping out unecessary columns  check the last two columns  +++++ ATTENTION +++++
    file_name = os.path.basename(csv_file)[:-4]  # -4 to remove the '.csv' from the name

    dfs_ready_to_transform[file_name + "_leaves"] = df_leaves # we want to consider the data as as without transforming

    df_level2 = convert_to_level2(df_leaves)
    dfs_ready_to_transform[file_name + "_level2"] = df_level2

    df_level1 = convert_to_level1(df_level2)
    dfs_ready_to_transform[file_name + "_level1"] = df_level1

    df_leaves.to_csv(f'{ready_to_transform_folder}/{file_name}_leaves.csv', index = False)
    df_level2.to_csv(f'{ready_to_transform_folder}/{file_name}_level2.csv', index = False)
    df_level1.to_csv(f'{ready_to_transform_folder}/{file_name}_level1.csv', index = False)


print (len(dfs_ready_to_transform))

# transfer to binary
for df_ready_to_transform_name, df_ready_to_transform in dfs_ready_to_transform.items():
    for threshold in exist_thresholds:
        df_transformed_to_binary = transform_to_binary(df_ready_to_transform, threshold)
        dfs_ready_to_process[df_ready_to_transform_name + f'_binary_{exist_thresholds}'] = df_transformed_to_binary
        df_transformed_to_binary.to_csv(f'{ready_to_process_folder}/{df_ready_to_transform_name}/{binary_folder}/_binary_{threshold}.csv')







# def transform_to_binary(df, threshold):
#     df_binary = df.applymap(lambda x: 1 if x > threshold else 0)
#     return df_binary




#     for exist_threshold in exist_thresholds:
#         df_binary = transform_to_binary(df, exist_threshold)
#         df_ready_array.append(df_binary)


# for df in df_ready_array:
#     print(df)


# file_path = 'data/qadc_multi.csv'
# # file_path = 'data/discogem_multi.csv'


# # CORR_COEFFICIENT = 'pearson'
# CORR_COEFFICIENT = 'spearman'
# # CORR_COEFFICIENT = 'kendall'




# critical_corr_values = ''

# df = pd.read_csv(file_path)

# # considering the sense columns only
# df = df.iloc[:, 8:]  

# plt.figure(figsize=(10,10))








# # creating the correlation matrix
# corr_matrix = df.corr(CORR_COEFFICIENT)



# corr_matrix.to_csv(f'results/{CORR_COEFFICIENT}_correlation_{file_path[5:]}')

# # creating a haet map
# ax = sns.heatmap(corr_matrix, annot=False, fmt=".2f", cmap='coolwarm', 
#             cbar=True, vmin=-1, vmax=1, linewidths=1, linecolor='black')

# # finding the max and min value
# numpy_corr_matrix = corr_matrix.to_numpy()
# np.fill_diagonal(numpy_corr_matrix, np.nan)
# flattened_corr_matrix = numpy_corr_matrix.flatten()
# max_corr_value = np.nanmax(flattened_corr_matrix)
# min_corr_value = np.nanmin(flattened_corr_matrix)

# # only annotating cells that are beyond the threshold, and the max/min values
# for i in range(corr_matrix.shape[0]):
#     for j in range(corr_matrix.shape[1]):
#         value = corr_matrix.iloc[i, j]
#         if value >= UPPER_CORR_THRESHOLD or value == max_corr_value or value <= LOWER_CORR_THRESHOLD or value == min_corr_value:
#             text = ax.text(j + 0.5, i + 0.5, f'{value:.2f}',
#                            ha="center", va="center", color="black")
#             col_i = corr_matrix.columns[i]
#             col_j = corr_matrix.columns[j]
#             critical_corr_values = critical_corr_values + f"{col_i} X {col_j}  = {round(value,3)}\n"

            
# # Customize the color bar
# cbar = ax.collections[0].colorbar
# cbar.set_ticks(np.arange(-1, 1.1, 0.1))  # Setting ticks from -1 to 1 at intervals of 0.1
# cbar.set_ticklabels([f'{tick:.1f}' for tick in np.arange(-1, 1.1, 0.1)])  # Formatting tick labels

# print(critical_corr_values)
# with open(f'results/correlation_{CORR_COEFFICIENT}_{file_path[5:-4]}.txt', 'w') as f:
#     f.write('Correlation Critical Values \n\n')
#     f.write(critical_corr_values)

# plt.title(f'Correlation {CORR_COEFFICIENT} Matrix')
# plt.savefig(f'results/correlation_{CORR_COEFFICIENT}_heatmap_{file_path[5:-4]}.png')
# plt.show()


# # for i in range (5):
# #     for j in range (5):
# #         # Access the first two columns by index
# #         first_column = df.iloc[:, i]  # This is the first column
# #         second_column = df.iloc[:, j]  # This is the second column
# #         # Create a scatter plot
# #         plt.scatter(first_column, second_column)
# #         plt.xlim(0, 1)
# #         plt.ylim(0, 1)
# #         plt.show()









# # col_pair = df.iloc[:,[0,1]]
# # print (col_pair)

# # corr_matrix = col_pair.corr()
# # print(corr_matrix)

# # corr_value = round(corr_matrix.iloc[0,1],3)
# # print(corr_value)

# # print("here")
# # first_col = df.iloc[:,0]
# # corr_with_1col = df.corrwith(first_col)
# # print(type(corr_with_1col))