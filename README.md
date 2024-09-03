# Statistical Analysis of Semantic Sense Pairs in Discourse Annotations

## Overview

This project performs a statistical analysis of semantic sense pairs in discourse annotations using various statistical tests such as:
  Chi-squared, 
  Fisher's Exact Test, 
  Odds Ratios (OR), 
  Pointwise Mutual Information (PMI), 
  and Yule's Q coefficient. 
The goal is to identify commonly confused sense pairs based on the PDTB-3 Sense Hierarchy levels.

## Project Structure

- **config.yaml**: Configuration file with paths, constants, and hyperparameters used in the analysis.
- **data_processing.py**: Functions for data transformation, removing all-zero columns, constructing contingency tables, and grouping data.
- **file_management.py**: Manages file operations, including directory creation, file cleaning, and saving results.
- **main.py**: The main script orchestrating the workflow, from data import and cleaning to statistical analysis.
- **requirements.txt**: Lists the Python dependencies required for the project.
- **statistical_analysis.py**: Implements statistical tests to analyze sense pairs and determine their associations.
- **Statistical Analysis of Semantic Sense Pairs in Discourse Annotations.pdf**: The detailed project report.

## Getting Started

### Prerequisites

Ensure you have Python installed on your system. You can download it from [python.org](https://www.python.org/).

### Installation

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/NawarTurk/SemanticSenseAnalysis.git
    cd SemanticSenseAnalysis/software/src
    ```

2. **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Configuration**:

    Edit the `config.yaml` file to ensure it has the correct paths and parameters for your environment.

## Running the Project

To run the project, execute the `main.py` script:

```bash
python main.py
```

## Results

The results of the analysis, including Chi-squared and Fisher's Exact Test P-values, Odds Ratios (OR), and PMI values, are stored in the `3_results` directory. Detailed contingency tables, CSV files with value matrices, and summary reports are available to help interpret the associations between sense pairs.
