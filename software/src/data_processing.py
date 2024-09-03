import pandas as pd
import yaml
import os

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

# Load paths from YAML config
result_folder = config['paths']['result_folder']
binary_folder = config['paths']['binary_folder']
contingency_table_folder = config['paths']['contingency_table_folder']


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
        try:
            file_path = os.path.join(result_folder, binary_folder, contingency_table_folder, f'{df_name}.txt')
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'a') as file:
                file.write(contingency_table_str + '\n\n' + '-.-.-.-.-.-.-.-.-.-\n')
        except IOError as e:
            print(f"An error occurred while writing to the file: {e}")
            
    return contingency_table




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
