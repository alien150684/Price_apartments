# Импорт стандартных библиотек Python
import glob
import os
import warnings
import joblib
import pickle
import gc
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import List

# Импорт библиотек для работы с данными
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from imblearn.over_sampling import RandomOverSampler, SMOTE
from lightgbm import LGBMClassifier
from scipy.sparse import load_npz
from scipy.stats import boxcox
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (auc, classification_report, confusion_matrix,
                             f1_score, get_scorer, log_loss, mean_absolute_error,
                             mean_squared_error, precision_recall_curve, r2_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, KFold, RandomizedSearchCV,
                                     train_test_split, cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler, OneHotEncoder,
                                   OrdinalEncoder, StandardScaler)
from tqdm import tqdm
from xgboost import XGBClassifier, XGBRegressor
from zipfile import ZipFile

# Импорт функции display из IPython.display
from IPython.display import display

# Управление предупреждениями
warnings.filterwarnings('ignore')

import pandas as pd
import joblib
import json
from tqdm import tqdm

import os
import pandas as pd

import os
import pandas as pd
import joblib

import os
import pandas as pd
import requests
import gdown

def load_data(path: str):
    """
    Загружает данные из файла или URL, анализируя его расширение или формат.

    :param path: путь к файлу или URL.
    :return: Загруженные данные (возможно, в формате DataFrame).
    """
    loaders = {
        'csv': pd.read_csv,
        'joblib': joblib.load,
        'pkl': pd.read_pickle,
        'json': pd.read_json,
        'txt': lambda f: open(f).read(),
        'xlsx': pd.read_excel
    }

    if path.startswith("https://docs.google.com/"):
        output = 'temp.xlsx'
        gdown.download(path, output, quiet=False)
        path = output
    elif path.startswith("http"):
        output = 'temp.xlsx'
        response = requests.get(path)
        with open(output, 'wb') as f:
            f.write(response.content)
        path = output

    extension = os.path.splitext(path)[1][1:]
    if extension not in loaders:
        raise ValueError(f"Unsupported file extension: {extension}")
    
    data = loaders[extension](path)
    
    return data


def process_data(data, method=None, constant_value=None, visualize=True, remove_duplicates=False):
    """
    Обрабатывает данные: устраняет пропущенные значения, удаляет дубликаты и визуализирует пропущенные данные.
    """
    
    def get_fill_values(data, columns):
        mode = data[columns].mode()
        mode_value = mode.iloc[0] if not mode.empty else data[columns].mean(numeric_only=True)
        
        return {
            'mean': data[columns].mean(numeric_only=True),
            'median': data[columns].median(numeric_only=True),
            'mode': mode_value,
            'constant': pd.Series(constant_value, index=columns),
            'interpolation': data[columns].select_dtypes(include='number').interpolate(method='linear')
        }

    columns_with_na = data.columns[data.isnull().any()].tolist()
    fill_values = get_fill_values(data, columns_with_na)

    if method == 'constant' and constant_value is None:
        raise ValueError("Provide 'constant_value' for the 'constant' method.")
    elif method == 'random':
        data[columns_with_na] = data[columns_with_na].apply(lambda col: col.fillna(np.random.choice(col.dropna())))
    elif method == 'dropna':
        data.dropna(inplace=True)
    elif method in fill_values:
        data.fillna(fill_values[method], inplace=True)
    elif method:
        raise ValueError(f"Unknown method: {method}")
    
    if remove_duplicates:
        print(f"Duplicates before removal: {data.duplicated().sum()}")
        data.drop_duplicates(inplace=True)
        print(f"Duplicates after removal: {data.duplicated().sum()}")
    
    if visualize:
        msno.matrix(data)
        plt.title('Missing Data Pattern Matrix', fontsize=16)
        plt.show()

    missing_values = data.isnull().sum()
    percent_missing = (missing_values / len(data)) * 100
    print(f"Missing values percentage:\n\n{percent_missing}\n")

def plot_outliers(data, columns=None, visualize=True, skip_no_outliers=True, 
                  sample_size=500, handle_outliers=None):
    """
    Визуализирует и выводит процент выбросов для каждой числовой переменной в DataFrame.

    Параметры:
    - data: DataFrame, исходные данные.
    - columns: список столбцов для проверки на выбросы. Если None, будут выбраны все числовые столбцы.
    - visualize: bool, указывает на необходимость визуализации данных.
    - skip_no_outliers: bool, если True, столбцы без выбросов будут пропущены.
    - sample_size: int, количество выборок для визуализации.
    - handle_outliers: str, метод обработки выбросов ('replace' или 'remove'). Если None, выбросы не обрабатываются.

    Возвращает:
    - DataFrame с обработанными выбросами (если handle_outliers указан). Иначе None.
    """
    columns = columns or data.select_dtypes(include=['int', 'float']).columns.tolist()
    k = 2.5
    
    if handle_outliers == 'replace':
        for col in columns:
            q1, q3 = data[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound, upper_bound = q1 - k * iqr, q3 + k * iqr
            data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)

    elif handle_outliers == 'remove':
        for col in columns:
            q1, q3 = data[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound, upper_bound = q1 - k * iqr, q3 + k * iqr
            data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
    
    def calculate_outliers_percent(column_data):
        q1, q3 = column_data.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound, upper_bound = q1 - k * iqr, q3 + k * iqr
        outliers = column_data[(column_data < lower_bound) | (column_data > upper_bound)]
        return (len(outliers) / len(column_data)) * 100

    non_binary_columns = [col for col in columns if data[col].nunique() > 2]
    outliers_percents = {col: calculate_outliers_percent(data[col]) for col in non_binary_columns}

    if visualize:
        columns_to_plot = [col for col, percent in outliers_percents.items() if percent > 0]

        if columns_to_plot:
            fig, ax = plt.subplots(figsize=(12, 8))

            sns.boxplot(data=data[columns_to_plot], showfliers=False, ax=ax,
                        boxprops=dict(facecolor='orange', edgecolor='black', alpha=0.5),
                        whiskerprops=dict(color='black', linestyle='-'))

            adjusted_sample_size = min(sample_size, data.shape[0])
            sampled_data = data.sample(n=adjusted_sample_size)

            sns.stripplot(data=sampled_data[columns_to_plot], color='blue', alpha=0.3, jitter=0.2, size=4, ax=ax)

            ax.set_title('Checking Outliers')
            ax.set_xlabel('Variables')
            ax.set_ylabel('Values')

            step = 1 / (len(columns_to_plot) + 1)
            for i, column in enumerate(columns_to_plot):
                fig.text((i + 1) * step, 0.90, f'{outliers_percents[column]:.2f}%', 
                        ha='center', va='center', fontsize=10, color='red')

            for line in range(1, 10):
                ax.axhline(line, color='gray', linestyle='--', linewidth=0.5)

            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        else:
            print("No columns with outliers to visualize.")
    
    for col, outliers_percent in outliers_percents.items():
        if not skip_no_outliers or outliers_percent > 0:
            print(f"{col} - Outliers: {outliers_percent:.2f}%")

    if handle_outliers:
        return data        

def plot_distributions_and_transform(data, columns=None, bins=80, max_categories=24, transform_method=None):
    transform_methods = {
        'boxcox': lambda x: boxcox(x + 1)[0],
        'log': np.log1p,
        'sqrt': np.sqrt,
        'cbrt': np.cbrt
    }

    if transform_method and transform_method not in transform_methods:
        raise ValueError(f"Invalid transform_method: {transform_method}")

    if columns is None:
        columns = data.columns

    feature_types = {
        'numeric': data[columns].select_dtypes(include=['int', 'float']).columns,
        'categorical': data[columns].select_dtypes(include=['object']).columns,
        'binary': [col for col in columns if data[col].nunique() == 2]
    }

    n_cols = min(3, sum(len(features) for features in feature_types.values()))
    fig, axs = plt.subplots((sum(len(features) for features in feature_types.values()) - 1) // n_cols + 1, n_cols, figsize=(15, 5 * ((sum(len(features) for features in feature_types.values()) - 1) // n_cols + 1)))
    axs = axs.ravel()

    i = 0
    for ftype, features in feature_types.items():
        for feature in features:
            if ftype == 'numeric':
                if transform_method:
                    data[feature] = transform_methods[transform_method](data[feature])
                sns.histplot(data[feature], bins=bins, kde=True, ax=axs[i])
            else:
                top_categories = data[feature].value_counts().index[:max_categories]
                data.loc[data[feature].isin(top_categories), feature].value_counts().plot(kind='bar', ax=axs[i], color='steelblue')
            axs[i].set_title(f'Распределение признака {feature}')
            i += 1

    for j in range(i, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    if transform_method:
        return data    

def plot_correlation_matrix(data, threshold=0.8):
    """
    Plot a correlation matrix for the given DataFrame. 
    If two features have a correlation higher than the threshold, drop the second feature.
    Parameters:
        data: pandas DataFrame.
        threshold: float, optional. 
            Features with a correlation higher than this value are considered highly correlated. 
            The second feature of each pair of highly correlated features will be dropped.
            Defaults to 0.8.
    Returns:
        new_data: a new DataFrame with highly correlated features dropped. 
                  If no such features are found, returns None.
    """
    # Calculate correlation matrix
    corr = data.corr().round(2)

    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Use a custom diverging colormap
    cmap = sns.diverging_palette(250, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

    plt.show()

    # Identify pairs of features that are highly correlated
    highly_correlated_features = set()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > threshold:
                featurename = corr.columns[i]
                highly_correlated_features.add(featurename)

    # Create a new DataFrame excluding highly correlated features
    if highly_correlated_features:
        new_data = data.drop(columns=highly_correlated_features)
        return new_data


def reduce_mem_usage(data):
    """ Функция для оптимизации использования памяти DataFrame (inplace). """
    
    # Расчет начального использования памяти -
    start_memory = data.memory_usage().sum() / 1024**2
    print(f"Initial memory usage: {start_memory:.2f} MB")
    
    # Создание словарей с диапазонами для каждого типа чисел
    int_type_dict = {
        (np.iinfo(np.int8).min,  np.iinfo(np.int8).max):  np.int8,
        (np.iinfo(np.int16).min, np.iinfo(np.int16).max): np.int16,
        (np.iinfo(np.int32).min, np.iinfo(np.int32).max): np.int32,
        (np.iinfo(np.int64).min, np.iinfo(np.int64).max): np.int64,
    }
    
    float_type_dict = {
        (np.finfo(np.float16).min, np.finfo(np.float16).max): np.float16,
        (np.finfo(np.float32).min, np.finfo(np.float32).max): np.float32,
        (np.finfo(np.float64).min, np.finfo(np.float64).max): np.float64,
    }
    
    # Обрабатываем каждый столбец в DataFrame
    for column in data.columns:
        col_type = data[column].dtype

        if np.issubdtype(col_type, np.integer):
            c_min = data[column].min()
            c_max = data[column].max()
            dtype = next((v for k, v in int_type_dict.items() if k[0] <= c_min and k[1] >= c_max), None)
            if dtype:
                data[column] = data[column].astype(dtype)
        elif np.issubdtype(col_type, np.floating):
            c_min = data[column].min()
            c_max = data[column].max()
            dtype = next((v for k, v in float_type_dict.items() if k[0] <= c_min and k[1] >= c_max), None)
            if dtype:
                data[column] = data[column].astype(dtype)
    
    # Расчет конечного использования памяти
    end_memory = data.memory_usage().sum() / 1024**2
    print(f"Final memory usage: {end_memory:.2f} MB")
    print(f"Reduced by {(start_memory - end_memory) / start_memory * 100:.1f}%")


class ProbabilityEncoder:
    def __init__(self, columns=None):
        self.columns = columns
        self.percentage_dicts = {}

    def fit(self, X, y):
        data = pd.concat([X, y], axis=1)
        self.columns = self.columns or [col for col in data.columns if col != y.name]
        self.target = y.name

        for column in self.columns:
            percentages = (data.groupby(column)[self.target].sum() / 
                           data.groupby(column)[self.target].count()) * 100
            self.percentage_dicts[column] = percentages.to_dict()

        return self

    def transform(self, X):
        return X.assign(**{col: X[col].map(self.percentage_dicts[col]).fillna(0) 
                           for col in self.columns})

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)
        
class DataPreprocessor:
    """
    Класс DataPreprocessor осуществляет предобработку данных, автоматически определяя числовые и категориальные признаки.

    Он применяет StandardScaler или MinMaxScaler к числовым признакам и OneHotEncoder к категориальным. Предобработка данных
    включает в себя обучение препроцессора на данных (метод fit), применение препроцессора к данным (метод transform) и
    комбинированный метод обучения и преобразования (метод fit_transform).
    """
    def __init__(self, scaling='scaler', encoding='onehot'):
        if scaling == 'scaler':
            self.num_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        elif scaling == 'MinMax':
            self.num_transformer = Pipeline(steps=[('scaler', MinMaxScaler())])

        if encoding == 'label':
            self.cat_transformer = Pipeline(steps=[('encoder', OrdinalEncoder())])
        elif encoding == 'onehot':
            self.cat_transformer = Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))])

        self.preprocessor = None

    def fit(self, X):
        numeric_features = X.select_dtypes(include=['int', 'float']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.num_transformer, numeric_features),
                ('cat', self.cat_transformer, categorical_features)
            ])

        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        return self.preprocessor.transform(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class ModelTrainer:
    def __init__(self, task_type='regression', balance=False, models=None, n_jobs=-1):
        self.task_type = task_type
        self.balance = balance
        self.n_jobs = n_jobs
        self.models = self._get_models(models)
        self.trained_models = {}
        self.score_func = roc_auc_score if task_type == 'classification' else mean_squared_error
        if balance and task_type == 'classification':
            self.sampler = SMOTE()

    def _get_models(self, models):
        regression_models = {
            'LinearRegression': LinearRegression(),
            'RandomForestRegressor': RandomForestRegressor(n_jobs=self.n_jobs),
            'GradientBoostingRegressor': GradientBoostingRegressor(),
            'CatBoostRegressor': CatBoostRegressor(silent=True),
            'LGBMClassifier': LGBMClassifier(silent=True,n_jobs=self.n_jobs)
        }

        classification_models = {
            'LogisticRegression': LogisticRegression(),
            'RandomForestClassifier': RandomForestClassifier(n_jobs=self.n_jobs),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'CatBoostClassifier': CatBoostClassifier(silent=True),
            'LGBMClassifier': LGBMClassifier(silent=True,n_jobs=self.n_jobs)
        }

        models_dict = classification_models if self.task_type == 'classification' else regression_models
        models = models_dict.keys() if models is None else [model for model in models if model in models_dict]
        return [models_dict[model_name] for model_name in models]

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        best_model, best_score = None, -float('inf')
        for model in self.models:
            try:
                if self.balance and self.task_type == 'classification':
                    X_train, y_train = self._apply_balance_strategy(model, X_train, y_train)
                model.fit(X_train, y_train)
                score = self._evaluate(model, X_test, y_test)
                self.trained_models[type(model).__name__] = model
                if score > best_score:
                    best_model, best_score = model, score
            except NotFittedError as e:
                print(f"Model {type(model).__name__} could not be fitted. Error: {str(e)}")
        print(f"\nBest model: {type(best_model).__name__}, with Score: {best_score}")
        return best_model, self.trained_models

    def _apply_balance_strategy(self, model, X_train, y_train):
        balance_strategies = {
            LogisticRegression: lambda x, y: (x, y, {'class_weight': 'balanced'}),
            RandomForestClassifier: lambda x, y: (x, y, {'class_weight': 'balanced'}),
            LGBMClassifier: lambda x, y: (x, y, {'class_weight': 'balanced'}),
            GradientBoostingClassifier: lambda x, y: (*self.sampler.fit_resample(x, y), {}),
            CatBoostClassifier: lambda x, y: (*self.sampler.fit_resample(x, y), {})
        }
        strategy = balance_strategies.get(type(model), lambda x, y: (x, y, {}))
        X_train, y_train, params = strategy(X_train, y_train)
        model.set_params(**params)
        return X_train, y_train

    def _evaluate(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        model_name = type(model).__name__
        score = self.score_func(y_test, y_pred)
        metric_name = 'ROC_AUC' if self.task_type == 'classification' else 'MSE'
        print(f"Model: {model_name}, {metric_name}: {score}")
        return score

class ModelOptimizer:
    def __init__(self, models=None, cv=5, scorer='roc_auc', balance=False, n_jobs=-1):
        self.cv = cv
        self.balance = balance
        self.n_jobs = n_jobs
        self.models = self._get_models(models)
        self.param_grids = self._get_param_grids()
        self.scorer = get_scorer(scorer)

    def _get_models(self, models):
        model_classes = {
            'LinearRegression': LinearRegression(),
            'RandomForestRegressor': RandomForestRegressor(n_jobs=self.n_jobs),
            'GradientBoostingRegressor': GradientBoostingRegressor(),
            'CatBoostRegressor': CatBoostRegressor(silent=True),
            'LogisticRegression': LogisticRegression(class_weight='balanced' if self.balance else None, n_jobs=self.n_jobs),
            'RandomForestClassifier': RandomForestClassifier(n_jobs=self.n_jobs),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'CatBoostClassifier': CatBoostClassifier(silent=True),
            'LGBMClassifier': LGBMClassifier(silent=True),
            'XGBClassifier': XGBClassifier(),
            'XGBRegressor': XGBRegressor()
        }
        return [model_classes[model] for model in models] if models else list(model_classes.values())

    def _get_param_grids(self):
        n_estimators = [100, 200, 300]
        learning_rate = [0.01, 0.1, 1.0]
        max_depth_short = [None, 5, 10]
        max_depth_long = [3, 7, 9]
        min_samples_split = [2, 5, 10]
        subsample = [0.5, 0.7, 1.0]
        depth = [6, 8, 10]
        colsample_bytree = [0.6, 0.8, 1.0]

        return {
            'RandomForestRegressor': {
                "n_estimators": n_estimators,
                "max_depth": max_depth_short,
                "min_samples_split": min_samples_split
            },
            'GradientBoostingRegressor': {
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "subsample": subsample,
                "max_depth": max_depth_long
            },
            'CatBoostRegressor': {
                "iterations": n_estimators,
                "learning_rate": learning_rate,
                "depth": depth
            },
            'LogisticRegression': {
                "C": learning_rate,
                "penalty": ['l1', 'l2']
            },
            'RandomForestClassifier': {
                "n_estimators": n_estimators,
                "max_depth": max_depth_short,
                "min_samples_split": min_samples_split
            },
            'GradientBoostingClassifier': {
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "subsample": subsample,
                "max_depth": max_depth_long
            },
            'CatBoostClassifier': {
                "iterations": n_estimators,
                "learning_rate": learning_rate,
                "depth": depth
            },
            'LGBMClassifier': {
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "max_depth": max_depth_long[:-1]
            },
             'XGBClassifier': {  # Добавляем набор параметров для XGBClassifier
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "max_depth": max_depth_long,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree
            },
            'XGBRegressor': {  # Добавляем набор параметров для XGBRegressor
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "max_depth": max_depth_long,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree
            }
        }

    def _is_classification(self):
        return any(isinstance(model, (LogisticRegression, RandomForestClassifier, GradientBoostingClassifier, CatBoostClassifier)) for model in self.models)

    def optimize(self, X_train, y_train, models=None):
        self.models = models or self.models
        self.optimized_models = []
        best_score = float('-inf')

        for model in self.models:
            model_name = model.__class__.__name__
            print(f"Optimizing {model_name}...")
            
            param_grid = self.param_grids.get(model_name, {})
            grid_search = GridSearchCV(model, param_grid, cv=self.cv, scoring=self.scorer, n_jobs=self.n_jobs)
            
            try:
                grid_search.fit(X_train, y_train)
            except Exception as e:
                print(f"An error occurred while fitting {model_name}: {e}")
                continue

            params = grid_search.best_params_
            score = grid_search.best_score_
            print(f"Best parameters for {model_name}: {params}")
            print(f"Best score for {model_name}: {score}")

            self.optimized_models.append({"model": grid_search.best_estimator_, "params": params, "score": score})

            if score > best_score:
                self.best_model = grid_search.best_estimator_
                best_score = score

        return self.optimized_models
    def evaluate(self, X_test, y_test, models):
        for model in models:
            try:
                # Поскольку scorer принимает модель, X и y_true, мы передаем их напрямую
                score = self.scorer(model, X_test, y_test)
            except Exception as e:
                print(f"An error occurred while evaluating {type(model).__name__}: {e}")
                continue

            print(f"Test score for {type(model).__name__}: {score}\n")

def plot_model(X_test, y_test, model=None, visualize=True):
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    # Преобразование вероятностей в метки классов с порогом 0.5
    y_pred = np.where(y_pred_prob >= 0.5, 1, 0)

    print(f"\nROC AUC Score:\n{roc_auc:.4f}")
    df_classification_report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()

    if visualize:
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted class")
        plt.ylabel("True class")
        plt.show()

        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
    else:
        print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    print("\nClassification Report:")
    display(df_classification_report)
    print(f"{'=='*40}\n")