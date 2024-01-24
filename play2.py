import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

file_path = 'data/qadc_multi.csv'
# file_path = 'data/discogem_multi.csv'

df = pd.read_csv(file_path)
df = df.iloc[:, 8:]  

plt.figure(figsize=(10,10))

data_matrix = df.values
processed_matrix = np.zeros_like(data_matrix)

for i, row in enumerate(data_matrix):
    largest_indices = np.argsort(row)[-2:]  # Get indices of two largest values
    if row[largest_indices].sum() > 0.8:  # Check if their sum is more than 0.5
        processed_matrix[i, largest_indices] = 1  # Set them to 1

processed_df = pd.DataFrame(processed_matrix, columns=df.columns)
processed_df.to_csv('play.csv')

correlation_matrix = processed_df.corr()







ax = sns.heatmap(correlation_matrix, annot=False, fmt=".2f", cmap='coolwarm', 
            cbar=True, vmin=-1, vmax=1, linewidths=1, linecolor='black')

# finding the max and min value
numpy_corr_matrix = correlation_matrix.to_numpy()
np.fill_diagonal(numpy_corr_matrix, np.nan)
flattened_corr_matrix = numpy_corr_matrix.flatten()
max_corr_value = np.nanmax(flattened_corr_matrix)
min_corr_value = np.nanmin(flattened_corr_matrix)

# only annotating cells that are beyond the threshold, and the max/min values
for i in range(correlation_matrix.shape[0]):
    for j in range(correlation_matrix.shape[1]):
        value = correlation_matrix.iloc[i, j]
        if value >= 0.7 or value == max_corr_value or value <= -0.7 or value == min_corr_value:
            text = ax.text(j + 0.5, i + 0.5, f'{value:.2f}',
                           ha="center", va="center", color="black")
            col_i = correlation_matrix.columns[i]
            col_j = correlation_matrix.columns[j]
            # critical_corr_values = critical_corr_values + f"{col_i} X {col_j}  = {round(value,3)}\n"

            
# Customize the color bar
cbar = ax.collections[0].colorbar
cbar.set_ticks(np.arange(-1, 1.1, 0.1))  # Setting ticks from -1 to 1 at intervals of 0.1
cbar.set_ticklabels([f'{tick:.1f}' for tick in np.arange(-1, 1.1, 0.1)])  # Formatting tick labels


plt.show()



# scatter plot between two features  -> results show no linear reltiaon
# for i in range (5):
#     for j in range (5):
#         # Access the first two columns by index
#         first_column = df.iloc[:, i]  # This is the first column
#         second_column = df.iloc[:, j]  # This is the second column
#         # Create a scatter plot
#         plt.scatter(first_column, second_column)
#         plt.xlim(0, 1)
#         plt.ylim(0, 1)
#         plt.show()









# col_pair = df.iloc[:,[0,1]]
# print (col_pair)

# corr_matrix = col_pair.corr()
# print(corr_matrix)

# corr_value = round(corr_matrix.iloc[0,1],3)
# print(corr_value)

# print("here")
# first_col = df.iloc[:,0]
# corr_with_1col = df.corrwith(first_col)
# print(type(corr_with_1col))