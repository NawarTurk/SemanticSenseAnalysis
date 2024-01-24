import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

file_path = 'data/qadc_multi.csv'
# file_path = 'data/discogem_multi.csv'

UPPER_CORR_THRESHOLD = 0.7
LOWER_CORR_THRESHOLD = -0.7
# CORR_COEFFICIENT = 'pearson'
CORR_COEFFICIENT = 'spearman'
# CORR_COEFFICIENT = 'kendall'




critical_corr_values = ''

df = pd.read_csv(file_path)

# considering the sense columns only
df = df.iloc[:, 8:]  

plt.figure(figsize=(10,10))

# creating the correlation matrix
corr_matrix = df.corr(CORR_COEFFICIENT)



corr_matrix.to_csv(f'results/{CORR_COEFFICIENT}_correlation_{file_path[5:]}')

# creating a haet map
ax = sns.heatmap(corr_matrix, annot=False, fmt=".2f", cmap='coolwarm', 
            cbar=True, vmin=-1, vmax=1, linewidths=1, linecolor='black')

# finding the max and min value
numpy_corr_matrix = corr_matrix.to_numpy()
np.fill_diagonal(numpy_corr_matrix, np.nan)
flattened_corr_matrix = numpy_corr_matrix.flatten()
max_corr_value = np.nanmax(flattened_corr_matrix)
min_corr_value = np.nanmin(flattened_corr_matrix)

# only annotating cells that are beyond the threshold, and the max/min values
for i in range(corr_matrix.shape[0]):
    for j in range(corr_matrix.shape[1]):
        value = corr_matrix.iloc[i, j]
        if value >= UPPER_CORR_THRESHOLD or value == max_corr_value or value <= LOWER_CORR_THRESHOLD or value == min_corr_value:
            text = ax.text(j + 0.5, i + 0.5, f'{value:.2f}',
                           ha="center", va="center", color="black")
            col_i = corr_matrix.columns[i]
            col_j = corr_matrix.columns[j]
            critical_corr_values = critical_corr_values + f"{col_i} X {col_j}  = {round(value,3)}\n"

            
# Customize the color bar
cbar = ax.collections[0].colorbar
cbar.set_ticks(np.arange(-1, 1.1, 0.1))  # Setting ticks from -1 to 1 at intervals of 0.1
cbar.set_ticklabels([f'{tick:.1f}' for tick in np.arange(-1, 1.1, 0.1)])  # Formatting tick labels

print(critical_corr_values)
with open(f'results/correlation_{CORR_COEFFICIENT}_{file_path[5:-4]}.txt', 'w') as f:
    f.write('Correlation Critical Values \n\n')
    f.write(critical_corr_values)

plt.title(f'Correlation {CORR_COEFFICIENT} Matrix')
plt.savefig(f'results/correlation_{CORR_COEFFICIENT}_heatmap_{file_path[5:-4]}.png')
plt.show()













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