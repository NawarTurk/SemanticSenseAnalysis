import os
import yaml

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
result_folder = config['paths']['result_folder']
binary_folder = config['paths']['binary_folder']
expected_table_folder = config['paths']['expected_table_folder']
analysis_value_matrices_folder = config['paths']['analysis_value_matrices_folder']
csv_files_folder = config['paths']['csv_files_folder']
summary_report_folder = config['paths']['summary_report_folder']

# Load constants from YAML config
chi2_squared_test = config['constants']['chi2_squared_test']
fisher_exact_test = config['constants']['fisher_exact_test']
yule_Q_test = config['constants']['yule_Q_test']
proposed_indicator = config['constants']['proposed_indicator']
conditional_probability = config['constants']['conditional_probability']
OR_ratio = config['constants']['OR_ratio']
pointwise_mutual_info = config['constants']['pointwise_mutual_info']

# Load hyperparameters from YAML config
report_critical_p_value = config['hyperparameters']['report_critical_p_value']
report_critical_upper_yule_q = config['hyperparameters']['report_critical_upper_yule_q']
report_critical_lower_yule_q = config['hyperparameters']['report_critical_lower_yule_q']
report_critical_upper_proposed_indicator = config['hyperparameters']['report_critical_upper_proposed_indicator']
report_critical_lower_proposed_indicator = config['hyperparameters']['report_critical_lower_proposed_indicator']
report_critical_conditional_probability = config['hyperparameters']['report_critical_conditional_probability']
report_critical_OR_value = config['hyperparameters']['report_critical_OR_value']
report_critical_upper_pointwise_mutual_info = config['hyperparameters']['report_critical_upper_pointwise_mutual_info']
report_critical_lower_pointwise_mutual_info = config['hyperparameters']['report_critical_lower_pointwise_mutual_info']


def create_required_directories():
    # List of directories to create
    base_data_folder = '..'  

    directories = [
        os.path.join(base_data_folder, 'data', '1_ready_to_transform'),
        os.path.join(base_data_folder, 'data', '2_ready_to_process', 'binary'),
        os.path.join(base_data_folder, 'data', '3_results', 'binary', 'contingency_tables'),
        os.path.join(base_data_folder, 'data', '3_results', 'binary', 'expected_tables'),
        os.path.join(base_data_folder, 'data', '3_results', 'summary_report')
    ]


    # Create each directory if it doesn't exist
    for dir in directories:
        full_path = os.path.join(dir)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            print(f"Created directory: {full_path}")
        else:
            print(f"Directory already exists: {full_path}")

def clean_files_within_directory(directory_name):
    """
    Clean all the files of the previous run to generate new results

    Parameters:
    directory_name (str): The name of the directory, relative to the script's location, to be cleaned
    """
    # Construct the full path to the directory relative to the current script
    directory_path = os.path.join(os.getcwd(), directory_name)

    if not os.path.exists(directory_path):
        print(f"Directory {directory_name} does not exist.")
        return

    # Walk through the directory
    try:
        for root, dirs, files in os.walk(directory_path):
            # Remove each file in the root and subdirectories
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
    except Exception as e:
            print(f"Error while cleaning files in {directory_name}: {e}")


def save_expected_table(expected_table, df_name):
    try:
        with open(os.path.join(result_folder, binary_folder, expected_table_folder, f'{df_name}.txt'), 'a') as file:
            expected_table_str = str(expected_table)
            file.write(expected_table_str + '\n\n' + '-.-.-.-.-.-.-.-.-.-\n')
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")  


def generate_csv(matrix_value_df, title):
    """
    Saves a DataFrame to a CSV file, ensuring the directory exists.

    Parameters:
    - matrix_value_df (pd.DataFrame): The DataFrame to save.
    - title (str): The title to use for the saved CSV file.
    """
    directory = os.path.join(result_folder, binary_folder, analysis_value_matrices_folder, csv_files_folder)
    file_path = os.path.join(directory, f'{title}.csv')

    if not os.path.exists(directory):
        os.makedirs(directory)  # Create any necessary intermediate directories

    try:
        matrix_value_df.to_csv(file_path)
    except Exception as e:
        print(f"An error occurred while saving CSV: {e}")

def generate_summary_report(df, df_name, method_used):
    result = ''
    positive_association = ''
    negative_association = ''

    if method_used == chi2_squared_test:
        result += (f'*** {df_name} ***\n\n')
        for row_label, row in df.iterrows():
            for col_label, value in row.items():
                if row_label != col_label:
                    if isinstance(value, (int, float)):
                        if value < report_critical_p_value:
                            result += (f'{row_label:<20} | {col_label:<20} | {round(value, 4):<7} | \n')
        result += ('______________________________\n\n\n')
        file_path = os.path.join(result_folder, summary_report_folder, f'{chi2_squared_test}.txt')
        write_to_file(file_path, result)

    elif method_used == fisher_exact_test:
        result += (f'*** {df_name} ***\n\n')
        for row_label, row in df.iterrows():
            for col_label, value in row.items():
                if row_label != col_label:
                    if isinstance(value, (int, float)):
                        if value < report_critical_p_value:
                            result += (f'{row_label:<20} | {col_label:<20} | {round(value, 4):<7} | \n')
        result += ('______________________________\n\n\n')
        file_path = os.path.join(result_folder, summary_report_folder, f'{fisher_exact_test}.txt')
        write_to_file(file_path, result)

    elif method_used == yule_Q_test:
        positive_association += (f'*** {df_name} ***\n\n Positive Association \n')
        negative_association += (f'*** {df_name} ***\n\n Negative Association \n')
        for row_label, row in df.iterrows():
            for col_label, value in row.items():
                if row_label != col_label:
                    if isinstance(value, (int, float)):
                        if round(value, 4) >= report_critical_upper_yule_q:
                            positive_association += (f'{row_label:<20} | {col_label:<20} | {round(value, 4):<7} | \n')
                        elif value <= report_critical_lower_yule_q:
                            negative_association += (f'{row_label:<20} | {col_label:<20} | {round(value, 4):<7} | \n')
        positive_association += ('______________________________\n\n\n')
        negative_association += ('______________________________\n\n\n')
        file_path = os.path.join(result_folder, summary_report_folder, f'{yule_Q_test}.txt')
        write_to_file(file_path, positive_association)
        write_to_file(file_path, negative_association)
    
    elif method_used == proposed_indicator:
        positive_association += (f'*** {df_name} ***\n\n Positive Association \n')
        negative_association += (f'*** {df_name} ***\n\n Negative Association \n')
        for row_label, row in df.iterrows():
            for col_label, value in row.items():
                if row_label != col_label:
                    if isinstance(value, (int, float)):
                        if value >= report_critical_upper_proposed_indicator:
                            positive_association += (f'{row_label:<20} | {col_label:<20} | {round(value, 4):<7} | \n')
                        elif value <= report_critical_lower_proposed_indicator:
                            negative_association += (f'{row_label:<20} | {col_label:<20} | {round(value, 4):<7} | \n')
        positive_association += ('______________________________\n\n\n')
        negative_association += ('______________________________\n\n\n')

        file_path = os.path.join(result_folder, summary_report_folder, f'{proposed_indicator}.txt')
        write_to_file(file_path, positive_association)
        write_to_file(file_path, negative_association)
  
    elif method_used == conditional_probability:
        positive_association += (f'*** {df_name} ***\n\n Positive Association\n')
        for row_label, row in df.iterrows():
            for col_label, value in row.items():
                if row_label != col_label:
                    if isinstance(value, (int, float)):
                        if value >= report_critical_conditional_probability:
                            positive_association += (f'{row_label:<20} | {col_label:<20} | {round(value, 4):<7} | \n')
        positive_association  += ('______________________________\n\n\n')
        file_path = os.path.join(result_folder, summary_report_folder, f'{conditional_probability}.txt')
        write_to_file(file_path, positive_association)

    elif method_used == OR_ratio:
        positive_association  += (f'*** {df_name} ***\n\n Positive Association \n')
        negative_association += (f'*** {df_name} ***\n\n Negative Association \n')
        for row_label, row in df.iterrows():
            for col_label, value in row.items():
                if row_label != col_label:
                    if isinstance(value, (int, float)):
                        if value >= report_critical_OR_value:
                            positive_association += (f'{row_label:<20} | {col_label:<20} | {round(value, 4):<7} | \n')
                        else:
                            negative_association += (f'{row_label:<20} | {col_label:<20} | {round(value, 4):<7} | \n')
        positive_association  += ('______________________________\n\n\n')
        negative_association += ('______________________________\n\n\n')
        file_path = os.path.join(result_folder, summary_report_folder, f'{OR_ratio}.txt')
        write_to_file(file_path, positive_association)
        write_to_file(file_path, negative_association)

    
    elif method_used == pointwise_mutual_info:
        result += (f'*** {df_name} ***\n\n')
        for row_label, row in df.iterrows():
            for col_label, value in row.items():
                if row_label != col_label:
                    if isinstance(value, (int, float)):
                        if value < report_critical_upper_pointwise_mutual_info and value > report_critical_lower_pointwise_mutual_info:
                            result += (f'{row_label:<20} | {col_label:<20} | {round(value, 4):<7} | \n')
        result += ('______________________________\n\n\n')
        file_name = f"{pointwise_mutual_info} NOTE THIS IS FOR REJECTION.txt"
        file_path = os.path.join(result_folder, summary_report_folder, file_name)
        write_to_file(file_path, result)

    else:
        print(f"Unknown method used: {method_used}")

def write_to_file(file_path, content):
    try:
        with open(file_path, 'a') as file:
            file.write(content)
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")