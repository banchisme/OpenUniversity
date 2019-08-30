import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import TransformerMixin
from collections import Counter, defaultdict
from university import exceptions


class NumericData(TransformerMixin):
    r"""fit and transform all numeric features

        example:
            num = NumericFeature
            num.fit_transform(X) # X is a pandas dataframe
    """
    def __init__(self, fillna='mean', normalize=True):
        r"""set parameters to deal with nemeric data

        """
        self.settings = {'fillna': fillna, 'normalize': normalize}
        self.parameters = {'order': defaultdict()}

    def fit(self, X):
        r"""
        :param X: pd.Seriese or a pd.DataFrame with one column
        :return: None
        """
        if self.settings['fillna'] == 'mean':
            self.parameters['fillna'] = X.mean()
        else:
            raise NotImplementedError

        if self.settings['normalize'] is True:
            self.parameters['normalize'] = {'mu': X.mean(), 'std': X.std()}

    def transform(self, X):
        X = X.copy()
        X = X.fillna(self.parameters['fillna'])

        if self.settings['normalize']:
            X = (X - self.parameters['normalize']['mu']) / self.parameters['normalize']['std']

        return X

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X)
        return self.transform(X)


class CategoricalData(TransformerMixin):
    r"""use one hot encoding to fit transform categorical feature

        example:
            cat = CategoricalFeature()
            y = cat.fit_transform(X) # X is a pandas dataframe
    """
    def __init__(self, encode='one-hot', most_common=10, handle_unknown='ignore', sparse=False, **kwargs):
        r"""encode categorical values
            Argument:
                encode (str): identifier for encoder, only one-host is supported now
                most_common (int): only most common n categories are kept
                handle_unknown (str): 'ignore' or 'error', parameter for the encoder
                sparse (bool): if True, return a sparse matrix
                kwargs (dict): kwargs for the encoder
            Return:
                a preprocessor that handle categorical values
        """
        if encode == 'one-hot':
            self.encoder = OneHotEncoder(handle_unknown=handle_unknown, sparse=sparse, **kwargs)
            self.settings = {'most_common': most_common}
            self.parameters = {}
        else:
            raise NotImplementedError

    def fit(self, X):
        r"""
        :param X (pd.DataFrame): data to be fitted, expected to be a pandas dataframe
        :return:
        """
        categories = self._get_most_common_categories(X)
        self.encoder.categories = categories
        self.encoder.fit(X)

    def transform(self, X):
        r"""encode X"""
        data = self.encoder.transform(X)
        columns = [X.columns[0] + '_' + category for category in self.encoder.categories[0]]
        return pd.DataFrame(data, index=X.index, columns=columns)

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X)
        return self.transform(X)

    def _get_most_common_categories(self, X):
        r""" return the most common type in X"""
        counter = Counter(X.copy().values.reshape(-1))
        return [[i for i, j in counter.most_common(self.settings['most_common'])]]


class OrdinalData(TransformerMixin):
    r"""encode ordinal data into integers"""
    def __init__(self, order, handle_unknown='ignore', fillna='most_common'):
        self.settings = {'handle_unknown': handle_unknown, 'fillna': fillna, 'order': order}
        order_dict = defaultdict(int)
        order_dict.update(dict(zip(order, range(len(order)))))
        self.parameters = {'order_dict': order_dict, 'order_category': order}

    def fit(self, X):
        if self.settings['fillna'] == 'most_common':
            self.parameters['fillna'] = self._get_most_common(X)
            self.parameters['order_dict'].default_factory = lambda: self.parameters['fillna']
        else:
            raise NotImplementedError

    def transform(self, X):
        X = X.fillna(self.parameters['fillna'])
        colname = X.columns[0]
        X[colname] = X[colname].map(self.parameters['order_dict'])
        return X

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X)
        return self.transform(X)

    def _get_most_common(self, X):
        r""" return the most common type in X"""
        counter = Counter(X.copy().values.reshape(-1))
        most_common_category = counter.most_common(1)[0][0]
        return self.parameters['order_category'].index(most_common_category)


class ColumnExtractor:
    r"""extract column as a pandas DataFrame"""
    def __init__(self, X, index_col):
        self.index_col = index_col
        self.X = X.copy()
        self._sanity_check()

    def extract(self, col):
        if type(self.index_col) is list:
            columns = self.index_col + [col]
        else:
            columns = [self.index_col, col]
        data = self.X[columns].set_index(self.index_col)
        return data

    def _sanity_check(self):
        if self.X[self.index_col].duplicated().sum() > 0:
            raise exceptions.DuplicateIndex(self.index_col)




