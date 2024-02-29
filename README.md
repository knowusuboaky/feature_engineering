# Feature Engineering

Unleash the full potential of your data with the Feature Engineering library, the ultimate Python toolkit designed to streamline and enhance your machine learning preprocessing and feature engineering workflows. Whether you're dealing with classification, regression, or any ML challenge, this library equips you with a robust set of tools to efficiently process numeric, categorical, and date features, tackle outliers, and engineer impactful new features.

## Further Description
Transform your machine learning workflows with Feature Engineering, the Python library designed to elevate your data preparation process. This cutting-edge tool streamlines the often cumbersome tasks of preprocessing and feature engineering, enabling you to unlock the full potential of your data with ease. Whether you're tackling classification, regression, or any other machine learning challenge, Feature Engineering equips you with a robust set of functionalities to efficiently handle numeric, categorical, and date features, manage outliers, and engineer new, impactful features.

Crafted with both novices and seasoned data scientists in mind, Feature Engineering offers an intuitive, flexible interface for custom transformations, advanced date processing, and dynamic feature selection strategies. From handling imbalances with sophisticated sampling techniques to optimizing your models through hyperparameter tuning, this library is your all-in-one solution for preparing your data for predictive modeling.

Dive into a world where data preprocessing and feature engineering are no longer barriers but catalysts for success. Elevate your machine learning projects with Feature Engineering and turn your data into a competitive advantage.

## Features
- **Data Preprocessing**: Simplify the often complex tasks of data cleaning, normalization, and transformation.
- **Feature Engineering**: Automatically extract and select the most relevant features for your models.
- **Handling Class Imbalance**: Utilize sophisticated sampling techniques to address class imbalances in your dataset.
- **Hyperparameter Tuning**: Optimize your machine learning models with integrated hyperparameter tuning capabilities.
- **Custom Transformations**: Apply custom data transformations with ease, tailored to your unique dataset requirements.

## Process Map
<img src="https://github.com/knowusuboaky/feature_engineering/blob/main/README%20file/mermaid%20figure%20-1.png?raw=true" width="1000" height="700" alt="Optional Alt Text">

## Installation

You can install the Scorecard Generator via pip:

``` bash

pip install feature_engineering==2.1.2
```

## Load Package
``` bash

from feature_engineering import FeatureEngineering
```


## Usage

``` bash
# Import necessary modules
import numpy as np
import pandas as pd

from feature_engineering import FeatureEngineering

# Load your dataset
data = pd.read_csv('your_dataset.csv')

# Deal with missing values in your target variable
from feature_engineering import Cleaner
cleaned_data = Cleaner(data, 
                      'target_column', 
                      method=['categorical', 'missing'])

# Initialize the FeatureEngineering object with dataset and configuration
FeatureEngineering =(data, target, numeric_columns=None, 
                     categorical_columns=None, date_columns=[], 
                     handle_outliers=None, imputation_strategy='mean', 
                     task='classification', n_features_to_select=None, 
                     threshold_correlation=0.9, filter_method=True, selection_strategy='embedded', 
                     sampling_strategies=None, include_holiday=None, custom_transformers=None, hyperparameter_tuning=None, evaluation_metric='accuracy', verbose=True, 
                     min_class_size=None, remainder = 'passthrough')

# Run the entire preprocessing and feature engineering pipeline
df_preprocessed, selected_features = FeatureEngineering.run()

# Display the preprocessed table/dataframe
df_preprocessed

# Display the selected features
selected_features

# Now your data is ready for model training
```

## Function Explanation

### Cleaner
The `Cleaner` function is designed for cleaning and optionally transforming a target variable within a dataset. It is flexible, allowing users to specify the treatment of the target variable based on its data type (`numerical` or `categorical`) and the desired action for handling missing values or transforming the data. The function parameters and the method options are explained below:

#### Function Parameters
`data` (pd.DataFrame): This is the dataset containing the target variable that you want to clean or transform. It should be passed as a pandas DataFrame.

`target_column` (str): This is the name of the column within the DataFrame that you intend to clean or transform. It specifies the target variable.

`method` (list): This is a two-element list where the first element specifies the data type, and the second element specifies the action for handling missing values or for transforming the data. The default is ['auto', 'auto'], indicating that both the data type and the action will be automatically determined based on the data.

#### Method Options
The `method` parameter is a list with two elements: [`data_type`, `action`].

First Element (`data_type`): Specifies the expected data type of the target column. Options include:

`'numerical'`: Indicates that the target variable is numeric (e.g., integers, floats).
`'categorical'`: Indicates that the target variable is categorical (e.g., strings, categories).
`'auto'`: The function will automatically infer the data type of the target variable based on its content. If the column's dtype is int64 or float64, it will be treated as numerical; otherwise, it will be treated as categorical.
Second Element (`action`): Specifies the action to take for handling missing values or transforming the target variable. Options vary depending on whether the target is `numerical` or `categorical`:

- **For numerical data**:
`'mean'`: Fills missing values with the mean of the column.
`'median'`: Fills missing values with the median of the column.
`'zero'`: Fills missing values with zero (0).
`'drop'`: Drops rows where the target column is missing.

**For categorical data**:
`'mode'`: Fills missing values with the mode (most frequent value) of the column.
`'missing'`: Fills missing values with the string 'Missing', explicitly marking them as missing.
`'encode'`: Applies one-hot encoding to the column, transforming it into multiple binary columns indicating the presence or absence of each category, including a separate column for missing values.
`'drop'`: Similar to numerical data, it drops rows where the target column is missing.

#### Examples of Method Picks
`method`=[`'numerical'`, `'mean'`]: For a `numerical target column`, this option will fill missing values with the column's `mean`. It's a common approach for handling missing data in numerical columns, assuming the data is roughly normally distributed or when the mean is considered a reasonable estimate for missing values.

`method`=[`'categorical'`, `'encode'`]: For a `categorical target column`, this option will perform `one-hot encoding`, creating a new binary column for each category (including missing values as a category). This transformation is useful for preparing categorical data for many types of machine learning models that require numerical input.

`method`=[`'auto'`, `'drop'`]: This option will automatically detect the data type of the target column and drop any rows where the `target column` is missing. This approach is data type-agnostic and can be used when preserving only complete cases is important for the analysis or modeling.

The flexibility of the method argument allows for tailored data preprocessing steps that can be adjusted based on the nature of the data and the requirements of subsequent analysis or modeling tasks.

### FeatureEngineering
Let's delve into detailed explanations of each argument in the `FeatureEngineering` class to understand their roles and implications in the feature engineering process:

#### Function Parameters
`data` (DataFrame): This is the primary dataset containing both `features` (independent variables) and the `target` (dependent variable). This DataFrame is the starting point of the feature engineering process, where all transformations, imputations, and selections will be applied.

`target` (string): The name of the column in data that represents the variable you are trying to predict. This column is excluded from transformations and is used to guide supervised learning tasks like classification or regression.

`numeric_columns` (list of strings, optional): A list specifying the columns in data that should be treated as numeric features. These columns are subject to scaling and numerical `imputation`. If not provided, the class automatically identifies columns of numeric data types (int64, float64) as numeric features.

`categorical_columns` (list of strings, optional): Specifies the columns in data that are categorical. These columns will undergo categorical `imputation` (for missing values) and `encoding` (transforming text or categorical labels into numeric form). If left unspecified, the class automatically selects columns of data types object and category as categorical.

`date_columns` (list of strings, optional): Identifies columns that contain date information. The class can expand these columns into multiple derived features such as `day of the month`, `month`, `year`, `day of the week`, and `week of the year`, enriching the dataset with potentially useful temporal information.

`handle_outliers` (list, optional): Dictates the approach for detecting and handling outliers in numeric data. The first element specifies the detection method (`'IQR'` for Interquartile Range or `'Z-score'`), and the second element defines the handling strategy (`'remove'` to exclude outliers, `'impute'` to replace outliers with a central tendency measure, or `'cap'` to limit outliers to a specified range).

`imputation_strategy` (string or dictionary, optional): Determines how missing values should be filled in. It can be a single string applicable to all columns (e.g., `'mean'`, `'median'`, `'most_frequent'`) or a dictionary mapping column names to specific imputation strategies, allowing for tailored imputation across different types of features.

`task` (string): Specifies the machine learning task, either `'classification'` or `'regression'`. This influences decisions related to model selection, feature selection techniques, and evaluation metrics appropriate for the predictive modeling task at hand.

`n_features_to_select` (int, optional): The number of features to retain after feature selection. If not set, no limit is applied to the number of features selected. This parameter is particularly useful when aiming to reduce dimensionality to a specific number of features.

`threshold_correlation` (float): Sets a threshold for identifying multicollinearity among features. Features with a pairwise correlation higher than this threshold are candidates for removal, helping to mitigate the adverse effects of multicollinearity on model performance.

`filter_method` (boolean): Enables or disables the application of filter methods in feature selection, such as removing features with `low variance` or `high correlation`. Setting this to `True` activates these filter methods, while `False` bypasses them.

`selection_strategy` (string): Chooses the strategy for feature selection. `'embedded'` refers to methods that integrate feature selection as part of the model training process (e.g., using model coefficients or importances), whereas `'wrapper'` involves selecting features by evaluating model performance across subsets of features.

`sampling_strategies` (list of dictionaries, optional): Outlines one or more strategies to address class imbalance through sampling. Each dictionary specifies a sampling technique (e.g., `'SMOTE'`, `'ADASYN'`) and its parameters. This allows for sophisticated handling of imbalanced datasets to improve model fairness and accuracy.

`include_holiday` (tuple, optional): If provided, adds a binary feature indicating whether a date falls on a public holiday. The tuple should contain the column name containing dates and the country code to reference the correct holiday calendar.

`custom_transformers` (dictionary, optional): Maps column names to custom transformer objects for applying specific transformations to selected columns. This enables highly customized preprocessing steps tailored to the unique characteristics of certain features.

`hyperparameter_tuning` (dictionary, optional): Specifies the configuration for hyperparameter tuning, including the tuning method (`'grid_search'` or `'random_search'`), parameter grid, and other settings like cross-validation folds. This facilitates the optimization of model parameters for improved performance.

`evaluation_metric` (string): The metric used to evaluate model performance during feature selection and hyperparameter tuning. The choice of metric should align with the machine learning task and the specific objectives of the modeling effort (e.g., `'accuracy'`, `'f1'`, `'r2'`).

`verbose` (boolean): Controls the verbosity of the class's operations. When set to True, progress updates and informational messages are displayed throughout the feature engineering process, offering insights into the steps being performed.

`min_class_size` (int, optional): Specifies the minimum size of the smallest class in a classification task. This parameter can influence the choice of sampling strategies and ensure that cross-validation splits are made in a way that respects the class distribution.

`remainder` (string, optional): Determines how columns not explicitly mentioned in `numeric_columns` or `categorical_columns` are treated. `'passthrough'` includes these columns without changes, while `'drop'` excludes them from the processed dataset.

These arguments collectively offer extensive control over the feature engineering process, allowing users to tailor preprocessing, feature selection, and model optimization steps to their specific dataset and modeling goals.

#### Method Options
- **`data`**:
   - **Options**: Any `pandas.DataFrame` with the dataset for preprocessing.

- **`target`**:
   - **Options**: Column name (string) in `data` that represents the dependent variable.

- **`numeric_columns`**:
   - **Options**:
     - `None`: Auto-select columns with data types `int64` and `float64`.
     - List of strings: Specific column names as numeric features.

- **`categorical_columns`**:
   - **Options**:
     - `None`: Auto-select columns with data types `object` and `category`.
     - List of strings: Specific column names as categorical features.

- **`date_columns`**:
   - **Options**:
     - `[]`: No date processing.
     - List of strings: Column names with date information for feature expansion.

- **`handle_outliers`**:
   - **Options**:
     - `None`: No outlier handling.
     - `[method, strategy]`:
       - `method`: `'IQR'`, `'Z-score'`.
       - `strategy`: `'remove'`, `'impute'`, `'cap'`.

- **`imputation_strategy`**:
   - **Options**:
     - String: `'mean'`, `'median'`, `'most_frequent'`, `'constant'`.
     - Dictionary: Maps columns to specific imputation strategies.

- **`task`**:
   - **Options**:
     - `'classification'`
     - `'regression'`

- **`n_features_to_select`**:
   - **Options**:
     - `None`: No limit on the number of features.
     - Integer: Specific number of features to retain.

- **`threshold_correlation`**:
    - **Options**:
      - Float (0 to 1): Cutoff for considering features as highly correlated.

- **`filter_method`**:
    - **Options**:
      - `True`: Apply filter methods for feature selection.
      - `False`: Do not apply filter methods.

- **`selection_strategy`**:
    - **Options**:
      - `'embedded'`: Model-based feature selection.
      - `'wrapper'`: Performance-based feature selection.

- **`sampling_strategies`**:
    - **Options**:
      - `None`: No sampling for class imbalance.
      - List of dictionaries: Specifies sampling techniques and parameters.
      - Example:
      ```bash
      ## Set sampling_strategies = [
         {'name': 'SMOTE', 'random_state': 42, 'k_neighbors': 2}  # Example specifying custom parameters for SMOTE
      ```

- **`include_holiday`**:
    - **Options**:
      - `None`: No holiday indicator.
      - Tuple: `(date_column_name, country_code)` for holiday feature.

- **`custom_transformers`**:
    - **Options**:
      - `{}`: No custom transformations.
      - Dictionary: Maps column names to custom transformer objects.
      - Example: 
      ```bash
      # class SquareTransformer(BaseEstimator, TransformerMixin):
         def fit(self, X, y=None):
            return self

         def transform(self, X):
            return X ** 2

      ##Now set custom_transformers = {'numeric_column': SquareTransformer()}
      ```
      - Another Example:
      ```bash
      class LogTransformer(BaseEstimator, TransformerMixin):
         def fit(self, X, y=None):
            return self  # nothing to do here

         def transform(self, X):
            # Ensure X is a DataFrame
            X = pd.DataFrame(X)
            # Apply transformation only to numeric columns
            for col in X.select_dtypes(include=['float64', 'int64']).columns:
                  # Ensure no negative values or zeros; you might adjust this logic based on your needs
                  X[col] = np.log1p(np.maximum(0, X[col]))
            return X
      ##Now set custom_transformers = {'numeric_column': LogTransformer()}
      ```

- **`hyperparameter_tuning`**:
    - **Options**:
      - `None`: No hyperparameter tuning.
      - Dictionary: Specifies tuning method, parameter grid, and settings.
      - Example: 
      ```bash
      ## Set hyperparameter_tuning = {
      'method': 'random_search',  # Choose 'grid_search' or 'random_search'
      'param_grid': {  # Specify the hyperparameter grid or distributions
         'n_estimators': [100, 200, 300],
         'max_depth': [5, 10, 15, None],
      },
      'cv': 5,  # Number of cross-validation folds
      'random_state': 42,  # Seed for reproducibility
      'n_iter': 10  # Number of parameter settings sampled (for RandomizedSearchCV)
   }
      ```

- **`evaluation_metric`**:
    - **Options**:
      - Classification: `'accuracy'`, `'precision'`, `'recall'`, `'f1'`, etc.
      - Regression: `'r2'`, `'neg_mean_squared_error'`, `'neg_mean_absolute_error'`, etc.

- **`verbose`**:
    - **Options**:
      - `True`: Display progress and informational messages.
      - `False`: Suppress messages.

- **`min_class_size`**:
    - **Options**:
      - `None`: Auto-determined.
      - Integer: Specifies the minimum size of any class.

- **`remainder`**:
    - **Options**:
      - `'passthrough'`: Include unspecified columns without changes.
      - `'drop'`: Exclude these columns from the processed dataset.

## Ideal Uses of the Feature Engineering Library

The Feature Engineering library is crafted to significantly enhance machine learning workflows through sophisticated preprocessing and feature engineering capabilities. Here are some prime scenarios where this library shines:

### Automated Data Preprocessing
- **Data Cleaning**: Automates the process of making messy datasets clean, efficiently handling missing values, outliers, and incorrect entries.
- **Data Transformation**: Facilitates seamless application of transformations such as scaling, normalization, or tailored transformations to specific data distributions.

### Feature Extraction and Engineering
- **Date Features**: Extracts and engineers meaningful features from date and time columns, crucial for time-series analysis or models relying on temporal context.
- **Text Data**: Engineers features from text data, including sentiment scores, word counts, or TF-IDF values, enhancing the dataset's dimensionality for ML algorithms.

### Handling Categorical Data
- **Encoding**: Transforms categorical variables into machine-readable formats, using techniques like one-hot encoding, target encoding, or embeddings.
- **Dimensionality Reduction**: Applies methods to reduce the dimensionality of high-cardinality categorical features, aiming to improve model performance.

### Dealing with Class Imbalance
- **Resampling Techniques**: Implements under-sampling, over-sampling, and hybrid methods to tackle class imbalance, enhancing model robustness.
- **Custom Sampling Strategies**: Allows for the experimentation with custom sampling strategies tailored to the dataset and problem specifics.

### Advanced Feature Selection
- **Filter Methods**: Employs variance thresholds and correlation matrices to eliminate redundant or irrelevant features.
- **Wrapper Methods**: Utilizes methods like recursive feature elimination to pinpoint the most predictive features.
- **Embedded Methods**: Leverages models with inherent feature importance metrics for feature selection.

### Model Optimization
- **Hyperparameter Tuning Integration**: Seamlessly integrates with hyperparameter tuning processes for simultaneous optimization of preprocessing steps and model parameters.
- **Pipeline Compatibility**: Ensures compatibility with scikit-learn pipelines, facilitating experimentation with various preprocessing and modeling workflows.

### Scalability and Flexibility
- **Custom Transformers**: Supports the creation and integration of custom transformers for unique preprocessing needs, offering unparalleled flexibility.
- **Scalability**: Designed to handle datasets of various sizes and complexities efficiently, from small academic datasets to large-scale industrial data.

### Interdisciplinary Projects
- **Cross-Domain Applicability**: Its versatile feature engineering capabilities make it suitable for a wide range of domains, including finance, healthcare, marketing, and NLP.

### Educational Use
- **Learning Tool**: Acts as an invaluable resource for students and professionals eager to delve into feature engineering and preprocessing techniques, offering hands-on experience.

### Research and Development
- **Experimental Prototyping**: Aids in the rapid prototyping of models within research settings, allowing researchers to concentrate on hypothesis testing and model innovation.

By providing a comprehensive suite of preprocessing and feature engineering tools, the Feature Engineering library aims to be an indispensable asset in enhancing the efficiency and efficacy of machine learning projects, democratizing advanced data manipulation techniques for practitioners across a spectrum of fields.

## Contributing
Contributions to the Feature Engineering are highly appreciated! Whether it's bug fixes, feature enhancements, or documentation improvements, your contributions can help make the library even more powerful and user-friendly for the community. Feel free to open issues, submit pull requests, or suggest new features on the project's GitHub repository.

## Documentation & Examples
For documentation and usage examples, visit the GitHub repository: https://github.com/knowusuboaky/feature_engineering

**Author**: Kwadwo Daddy Nyame Owusu - Boakye\
**Email**: kwadwo.owusuboakye@outlook.com\
**License**: MIT
