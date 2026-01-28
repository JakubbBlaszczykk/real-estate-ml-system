import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin



class AreaFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Combines multiple area-related features into a single unified feature called `MainArea`.

    This transformer creates a new column `MainArea` by selecting the first non-null value
    from a predefined list of area-related columns in a priority order. This allows the model
    to use the most reliable available area measurement for each property while reducing
    redundancy and missing values.

    The original area-related columns are removed after the transformation.

    Parameters
    ----------
    area_columns : list of str
        List of column names containing different area measurements. The order of this list
        defines the priority used to select the main area value.

    Input
    -----
    X : pandas.DataFrame
        Input dataset containing the area-related columns.

    Output
    ------
    X_transformed : pandas.DataFrame
        Dataset with a new column `MainArea` and without the original area-related columns.
    """
    def __init__(self, area_columns):
        self.area_columns = area_columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X["MainArea"] = X[self.area_columns].bfill(axis=1).iloc[:, 0]
        X = X.drop(columns=self.area_columns)
        return X
    

class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """
    Groups rare categories in categorical columns into a single 'Other' category.

    Categories whose relative frequency is below the specified threshold are replaced
    with the value 'Other'. This helps reduce high cardinality, noise and dimensionality
    after one-hot encoding.

    Parameters
    ----------
    columns : list of str
        List of categorical columns to process.

    min_frequency : float, default=0.01
        Minimum relative frequency required to keep a category. Categories with frequency
        below this threshold will be grouped into 'Other'.
    """

    def __init__(self, columns, min_frequency=0.01):
        self.columns = columns
        self.min_frequency = min_frequency
        self.frequent_categories_ = {}

    def fit(self, X, y=None):
        X = X.copy()

        for col in self.columns:
            freq = X[col].value_counts(normalize=True)
            self.frequent_categories_[col] = freq[freq >= self.min_frequency].index.tolist()

        return self

    def transform(self, X):
        X = X.copy()

        for col in self.columns:
            X[col] = X[col].where(X[col].isin(self.frequent_categories_[col]), other="Other")

        return X


class BooleanNormalizer(BaseEstimator, TransformerMixin):
    """
    Normalizes boolean-like categorical columns to numeric 0/1 values.

    This transformer converts common string and boolean representations such as:
    - "Yes", "No", "True", "False", "Y", "N"
    - True, False

    into numeric values:
    - 1 for True / Yes
    - 0 for False / No

    Missing values are preserved as NaN.

    Parameters
    ----------
    columns : list of str
        List of columns to normalize.
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        true_values = {"yes", "true", "1", 1, True}
        false_values = {"no", "false", "0", 0, False}

        for col in self.columns:
            X[col] = X[col].astype(str).str.lower().map(
                lambda x: 1 if x in true_values else (0 if x in false_values else np.nan)
            )

        return X


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts year and month features from a date column.

    This transformer parses a date column and creates two new numerical features:
    - <prefix>Year
    - <prefix>Month

    The original date column is removed after transformation.

    Invalid or missing dates are safely converted to NaN.

    Parameters
    ----------
    column : str
        Name of the date column to process.

    prefix : str, default="Publish"
        Prefix used for the generated feature names.
    """

    def __init__(self, column, prefix="Publish"):
        self.column = column
        self.prefix = prefix

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        dates = pd.to_datetime(X[self.column], errors="coerce")

        X[f"{self.prefix}Year"] = dates.dt.year
        X[f"{self.prefix}Month"] = dates.dt.month

        X = X.drop(columns=[self.column])

        return X


class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Clips numerical features to a specified quantile range to reduce the impact of outliers.

    For each selected column, values below the lower quantile and above the upper quantile
    are clipped to the respective boundary values.

    This transformer does not remove rows and does not create missing values.

    Parameters
    ----------
    columns : list of str
        List of numerical columns to clip.

    lower_quantile : float, default=0.01
        Lower quantile used for clipping.

    upper_quantile : float, default=0.99
        Upper quantile used for clipping.
    """

    def __init__(self, columns, lower_quantile=0.01, upper_quantile=0.99):
        self.columns = columns
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.bounds_ = {}

    def fit(self, X, y=None):
        X = X.copy()

        for col in self.columns:
            lower = X[col].quantile(self.lower_quantile)
            upper = X[col].quantile(self.upper_quantile)
            self.bounds_[col] = (lower, upper)

        return self

    def transform(self, X):
        X = X.copy()

        for col, (lower, upper) in self.bounds_.items():
            X[col] = X[col].clip(lower, upper)

        return X


class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Drops specified columns from the dataset.

    This transformer removes a predefined list of columns from the input DataFrame.
    It is useful for removing unused, redundant or intermediate features inside
    a preprocessing pipeline.

    Parameters
    ----------
    columns : list of str
        List of column names to drop.
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.drop(columns=self.columns, errors="ignore")
        return X
