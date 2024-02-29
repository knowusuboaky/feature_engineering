import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, RFECV, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
from imblearn.combine import SMOTEENN, SMOTETomek
from collections import Counter
from IPython.display import display
import ipywidgets as widgets
from datetime import datetime
import holidays
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer



def clean_and_transform_target(data, target_column, method=['auto', 'auto']):
    """
    Cleans and potentially transforms a target variable in a dataset based on a specified method.
    The method includes both the data type and the action for handling missing values or transformations.

    Parameters:
    - data (pd.DataFrame): The dataset containing the target variable.
    - target_column (str): The name of the target variable column.
    - method (list): A two-element list where the first element specifies the data type
                     ('numerical', 'categorical', 'auto') and the second element specifies
                     the action ('mean', 'median', 'mode', 'drop', 'encode', 'missing', 'zero').

    Returns:
    - pd.DataFrame: Dataset with the cleaned or transformed target variable.
    """
    data_type, action = method
    
    if data_type == 'auto':
        if data[target_column].dtype in [np.int64, np.float64]:
            data_type = 'numerical'
        else:
            data_type = 'categorical'
    
    if data_type == 'numerical':
        if action == 'mean':
            fill_value = data[target_column].mean()
        elif action == 'median':
            fill_value = data[target_column].median()
        elif action == 'zero':
            fill_value = 0
        elif action == 'drop':
            return data.dropna(subset=[target_column])
        data[target_column].fillna(fill_value, inplace=True)
    
    elif data_type == 'categorical':
        if action == 'mode':
            fill_value = data[target_column].mode()[0]
        elif action == 'missing':
            fill_value = 'Missing'
        elif action == 'drop':
            return data.dropna(subset=[target_column])
        elif action == 'encode':
            # Perform one-hot encoding on the target variable
            data = pd.get_dummies(data, columns=[target_column], prefix=target_column, dummy_na=True)
            return data
        data[target_column].fillna(fill_value, inplace=True)
    
    return data



class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self, df, target, numeric_columns=None, categorical_columns=None, date_columns=[], handle_outliers=None,
                 imputation_strategy='mean', task='classification', n_features_to_select=None, 
                 threshold_correlation=0.9, filter_method=True, selection_strategy='embedded', 
                 sampling_strategies=None, include_holiday=None, custom_transformers=None, 
                 hyperparameter_tuning=None, evaluation_metric='accuracy', verbose=True, min_class_size=None, remainder = 'passthrough'):
        
        
        # Initialization of parameters
        self.df = df
        self.target = target
        self.numeric_columns = numeric_columns if numeric_columns is not None else df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_columns = categorical_columns if categorical_columns is not None else df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.date_columns = date_columns
        self.imputation_strategy = imputation_strategy
        self.task = task
        self.n_features_to_select = n_features_to_select
        self.threshold_correlation = threshold_correlation
        self.filter_method = filter_method
        self.selection_strategy = selection_strategy
        self.sampling_strategies = sampling_strategies
        self.verbose = verbose
        self.X_preprocessed = None
        self.y = None
        self.selected_features = None
        self.hyperparameter_tuning = hyperparameter_tuning
        self.custom_transformers = custom_transformers if custom_transformers is not None else {}
        self.include_holiday = include_holiday
        self.evaluation_metric = evaluation_metric
        self.min_class_size = min_class_size
        self.remainder = remainder
        self.handle_outliers = handle_outliers


        # Progress bar setup
        style = {'description_width': 'initial', 'bar_color': '#00AEFF'}

        self.progress = widgets.FloatProgress(
        value=0,
        min=0,
        max=100,
        bar_style='',
        style=style,
        description='Initializing...',)

        self.__verbose = verbose

        if self.__verbose:
          display(self.progress)

        self.progress.description = 'Initializing...'
        self.progress.value = 0

        # Validate input data
        self.validate_input()


    def validate_input(self):
        """Validates the input DataFrame and target variable."""

        self.progress.description='Validating input data: ensuring target column exists and determining data types.'
        self.progress.value = 0
        
        assert self.target in self.df.columns, f"Target column '{self.target}' not found in DataFrame."
        self.numeric_columns = [col for col in self.numeric_columns if col != self.target]
        self.categorical_columns = [col for col in self.categorical_columns if col != self.target]

        self.progress.description='Input data validated successfully.'
        self.progress.value = 100


    def preprocess_data(self):
        """Preprocesses the data according to specified parameters."""

        self.progress.description='Preprocessing numeric columns: Applying imputation for missing values and scaling.'
        self.progress.value = 0

        # Check and replace infinite values with NaN
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Apply custom transformers before standard preprocessing
        for col, transformer in self.custom_transformers.items():
            if col in self.df.columns:
                self.df[col] = transformer.fit_transform(self.df[[col]])

        # Advanced Date column processing
        for date_col in self.date_columns:
            if date_col in self.df.columns:
                self.df[date_col] = pd.to_datetime(self.df[date_col], errors='coerce')
                
                # Basic date features
                self.df[f'{date_col}_day'] = self.df[date_col].dt.day
                self.df[f'{date_col}_month'] = self.df[date_col].dt.month
                self.df[f'{date_col}_year'] = self.df[date_col].dt.year
                self.df[f'{date_col}_dayofweek'] = self.df[date_col].dt.dayofweek
                self.df[f'{date_col}_weekofyear'] = self.df[date_col].dt.isocalendar().week
                
                # Advanced features
                # Check if the date is a public holiday
                if self.include_holiday:
                    country = self.include_holiday[1] if len(self.include_holiday) > 1 else 'UnitedStates'
                    country_holidays = getattr(holidays, country)()  # Dynamically get the holiday class
                    self.df[f'{date_col}_is_public_holiday'] = self.df[date_col].apply(lambda x: x in country_holidays).astype(int)
                
                # Cleanup: Drop the original date column
                self.df.drop(columns=[date_col], inplace=True)

                # Update numeric columns list
                self.numeric_columns.extend([
                    f'{date_col}_day', f'{date_col}_month', f'{date_col}_year', 
                    f'{date_col}_dayofweek', f'{date_col}_weekofyear'
                ])
                if self.include_holiday:
                    self.numeric_columns.append(f'{date_col}_is_public_holiday')

        # Outlier detection and handling
        if self.handle_outliers is not None:
            assert len(self.handle_outliers) == 2, "handle_outliers must be a list with two elements: ['method', 'strategy']."
            method, strategy = self.handle_outliers
            
            columns = self.numeric_columns  # Assuming outliers are to be handled in numeric columns
            for col in columns:
                if method == 'IQR':
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                elif method == 'Z-score':
                    mean = self.df[col].mean()
                    std = self.df[col].std()
                    lower_bound = mean - 3 * std
                    upper_bound = mean + 3 * std
                else:
                    raise ValueError(f"Unsupported outlier detection method: {method}")

                # Identify outliers
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]

                # Handle outliers
                if strategy == 'remove':
                    self.df = self.df[~self.df.index.isin(outliers.index)]
                elif strategy == 'impute':
                    self.df.loc[outliers.index, col] = self.df[col].median()  # Example: impute with median
                elif strategy == 'cap':
                    self.df.loc[self.df[col] < lower_bound, col] = lower_bound
                    self.df.loc[self.df[col] > upper_bound, col] = upper_bound


        # Conditional imputation logic
        # Ensure no duplicate columns between numeric and categorical lists
        self.numeric_columns = list(set(self.numeric_columns) - set(self.categorical_columns))
        
        # Conditional logic for imputation
        if self.imputation_strategy == 'IterativeImputer':
            numeric_transformer = Pipeline(steps=[
                ('imputer', IterativeImputer(random_state=0)),
                ('scaler', StandardScaler())])
            # Apply to all numeric columns
            transformers = [('num', numeric_transformer, self.numeric_columns)]
        else:
            transformers = []
            for col in self.numeric_columns:
                strategy = self.imputation_strategy[col] if isinstance(self.imputation_strategy, dict) else self.imputation_strategy
                transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy=strategy)),
                    ('scaler', StandardScaler())])
                transformers.append((col, transformer, [col]))

        # Categorical transformers, ensuring no duplicate handling of columns
        for col in self.categorical_columns:
            transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))])
            transformers.append((col, transformer, [col]))

        # Apply ColumnTransformer
        if self.remainder is not None:
            # If remainder is specified, use it in the ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder=self.remainder)  # Use the class attribute here
        else:
            # If remainder is None, omit it (default behavior is 'drop')
            preprocessor = ColumnTransformer(
                transformers=transformers)

        # Applying preprocessing
        self.y = self.df[self.target]
        X = self.df.drop(columns=[self.target], errors='ignore')
        X_preprocessed = preprocessor.fit_transform(X)
        feature_names_out = preprocessor.get_feature_names_out()
        self.X_preprocessed = pd.DataFrame(X_preprocessed, columns=feature_names_out, index=self.df.index)

        if self.filter_method:
            # Variance threshold
            selector = VarianceThreshold()
            self.X_preprocessed = pd.DataFrame(selector.fit_transform(self.X_preprocessed), 
                                               columns=self.X_preprocessed.columns[selector.get_support()],
                                               index=self.X_preprocessed.index)
            # Correlation filtering
            corr_matrix = self.X_preprocessed.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
            to_drop = [column for column in upper.columns if any(upper[column] > self.threshold_correlation)]
            self.X_preprocessed.drop(columns=to_drop, inplace=True)

        self.progress.description='Data preprocessed successfully.'
        self.progress.value = 100


    def apply_sampling_techniques(self):
        """Applies sampling techniques if specified to handle class imbalance."""

        self.progress.description='Applying sampling techniques to address data imbalance, if applicable.'
        self.progress.value = 0
    
        # Define available sampling strategies with default configurations
        sampling_strategies = {
            'SMOTE': SMOTE(),
            'ADASYN': ADASYN(),
            'RandomUndersampling': RandomUnderSampler(),
            'ClusterCentroids': ClusterCentroids(),
            'SMOTEENN': SMOTEENN(),
            'SMOTETomek': SMOTETomek()
        }

        # Iterate through specified sampling strategies and apply them
        if self.sampling_strategies:
            for strategy_spec in self.sampling_strategies:
                strategy_name = strategy_spec['name']
                if strategy_name in sampling_strategies:
                    # Create the sampler with custom parameters
                    sampler = sampling_strategies[strategy_name].set_params(**{k: v for k, v in strategy_spec.items() if k != 'name'})
                    X_resampled, y_resampled = sampler.fit_resample(self.X_preprocessed, self.y)
                    self.X_preprocessed = pd.DataFrame(X_resampled, columns=self.X_preprocessed.columns)
                    self.y = pd.Series(y_resampled)

        self.progress.description='Sampling applied successfully.'
        self.progress.value = 100


    def perform_hyperparameter_tuning(self, model, X, y):

        self.progress.description='Performing Hyperparameter Tuning...'
        self.progress.value = 0

        if not self.hyperparameter_tuning:
            return model  # No tuning requested

        tuning_method = self.hyperparameter_tuning.get('method', 'grid_search')
        param_grid = self.hyperparameter_tuning.get('param_grid', {})
        cv = self.hyperparameter_tuning.get('cv', 5)  # Default to 5-fold cross-validation
        random_state = self.hyperparameter_tuning.get('random_state', None)  # Default to None if not specified
        n_iter = self.hyperparameter_tuning.get('n_iter', 10)  # Default to 10 iterations for RandomizedSearchCV

        if tuning_method == 'grid_search':
            search = GridSearchCV(model, param_grid, cv=cv, scoring=self.evaluation_metric)
        elif tuning_method == 'random_search':
            search = RandomizedSearchCV(model, param_grid, cv=cv, n_iter=n_iter, random_state=random_state, scoring=self.evaluation_metric)
        else:
            raise ValueError("Unsupported tuning method specified.")
        
        search.fit(X, y)

        self.progress.description='Hyperparameter Tuning applied successfully.'
        self.progress.value = 100

        return search.best_estimator_
    

    def select_features(self):
        """Selects features based on the specified selection strategy."""

        self.progress.description='Performing feature selection: Identifying the most relevant features for the model...'
        self.progress.value = 0

        if self.min_class_size is not None:
            n_splits = min(5, self.min_class_size)
        else:
            # Calculate min_class_size and n_splits dynamically if not provided
            min_class_size = min(self.y.value_counts())  # Find the size of the smallest class
            n_splits = min(5, min_class_size)  # Ensure n_splits does not exceed the smallest class size
        

        if self.selection_strategy == 'embedded':
            model = self.select_embedded_method(n_splits=n_splits)
            selector = SelectFromModel(model).fit(self.X_preprocessed, self.y)       
        elif self.selection_strategy == 'wrapper':
            estimator = RandomForestClassifier() if self.task == 'classification' else RandomForestRegressor()
            selector = RFECV(estimator, step=1, cv=StratifiedKFold(n_splits=n_splits),
                             scoring=self.evaluation_metric, 
                             min_features_to_select=self.n_features_to_select or 1).fit(self.X_preprocessed, self.y)
        else:
            raise ValueError("Invalid selection strategy specified.")
        
        self.selected_features = self.X_preprocessed.columns[selector.get_support()]
        self.X_preprocessed = self.X_preprocessed.loc[:, self.selected_features]

        self.progress.description='Features selected successfully.'
        self.progress.value = 100


    def select_embedded_method(self, n_splits=5):
        # Assume 'model' is initialized based on the task (classification or regression)
        models = {
            'classification': {
                'DecisionTree': DecisionTreeClassifier(),
                'RandomForest': RandomForestClassifier()
            },
            'regression': {
                'LassoCV': LassoCV(cv=n_splits),
                'RidgeCV': RidgeCV(cv=n_splits),
                'ElasticNetCV': ElasticNetCV(cv=n_splits),
                'DecisionTree': DecisionTreeRegressor()
            }
        }[self.task]

        best_score, best_model = -np.inf, None
        for model in models.values():
            score = cross_val_score(model, self.X_preprocessed, self.y, cv=n_splits, scoring='accuracy' if self.task == 'classification' else 'r2').mean()
        
            # Now, apply hyperparameter tuning if specified
            tuned_model = self.perform_hyperparameter_tuning(model, self.X_preprocessed, self.y)

            if score > best_score:
                best_score, best_model = score, tuned_model
        return best_model    
    

    def run(self):
        """Executes the entire feature engineering process and ensures unique names for columns and features."""

        # Step 1: Preprocess data
        self.preprocess_data()

        # Step 2: Apply sampling techniques if specified
        if self.sampling_strategies:
            self.apply_sampling_techniques()

        # Step 3: Perform feature selection
        self.select_features()

        # Concatenate preprocessed features with the target variable for the final DataFrame
        df_final = pd.concat([self.X_preprocessed, self.df[[self.target]]], axis=1)

        self.progress.description = 'Feature engineering process completed successfully. Data is now ready for model training.'
        self.progress.value = 100

        # Return the preprocessed data and unique selected features
        return df_final, self.selected_features
