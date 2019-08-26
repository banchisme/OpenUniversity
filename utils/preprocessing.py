import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer, OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from scipy import stats
#from clickfeatures import regularity, procrastination, timeseries
import math
import scipy
import re
from sklearn.base import TransformerMixin
from collections import Counter
from utils import exceptions


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
        self.parameters = {}

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


def build_timebased_regularity(studentVle, assessments, using_testing_dates=0, user_defined_range=[]):
    # define helper functions
    def extract_timestamps_and_weights(df):
        return pd.Series({
            'timestamps': df['date'].tolist(),
            'weights': df['sum_click'].tolist()})

    def extract_test_dates(df, num_dates=3):
        return pd.Series({'dates': sorted(df['date'].tolist())[:num_dates]})

    def get_time_regularity(df, using_testing_dates=using_testing_dates, 
        user_defined_range=user_defined_range):
        if user_defined_range:
            start, end = user_defined_range
        elif 0 <= using_testing_dates <= 2:
            start = 0
            end = df['dates'][using_testing_dates]

        ts = df['timestamps']
        ws = df['weights']
        # cut off dates after the end date
        ts = filter(lambda x: x <= end, ts)
        ws = ws[:len(ts)]
        r = regularity.TimeRegularity(ts, ws, end=end, unit='day')
        return r.get_regularity()

    def unwrap_time_regularity(df, metrics=['pwd', 'ws1', 'ws2', 'ws3', 'fwd']):
        assert 'regularity' in df.columns
        for metric in metrics:
            df[metric] = df['regularity'].apply(
                lambda x: x[metric] if metric in x else np.nan)
        return df

    # make a copy
    studentVle = studentVle.copy()
    assessments = assessments.copy()

    #######################
    # preprocess studentVle
    #######################
    studentVle = studentVle.query('date >= 0')

    # aggregate daily clicks
    identifiers = ['code_module', 'code_presentation', 'id_student', 'date']
    studentVle = studentVle.groupby(
        identifiers)['sum_click'].agg('sum').reset_index()
    studentVle.sort_values(identifiers, inplace=True)

    # apply function
    identifiers = ['code_module', 'code_presentation', 'id_student']
    studentVle = studentVle.groupby(
        identifiers).apply(extract_timestamps_and_weights)
    studentVle.reset_index(inplace=True)

    #######################
    # preprocess assessments
    #######################
    identifiers = ['code_module', 'code_presentation']
    assessments = assessments.groupby(identifiers).apply(extract_test_dates)
    assessments.reset_index(inplace=True)

    # merge
    identifiers = ['code_module', 'code_presentation']
    studentVle = studentVle.merge(assessments, on=identifiers, how='left')

    # build features
    studentVle['regularity'] = studentVle.apply(get_time_regularity, axis=1)
    studentVle = unwrap_time_regularity(studentVle)

    # drop unuseful columns
    studentVle.drop([
        'regularity', 'timestamps', 'weights', 'dates'], axis=1, inplace=True)

    return studentVle


def build_activitybased_regularity(studentVle, vle, assessments, drop_columns=True,
    using_testing_dates=0, user_defined_range=[]):

    valid_activies = [
        'forumng', 'subpage', 'oucontent', 'homepage', 'quiz',
        'resource', 'url', 'ouwiki', 'externalquiz', 'page',
        'oucollaborate', 'questionnaire', 'ouelluminate',
        'glossary', 'dualpane', 'dataplus', 'folder',
        'sharedsubpage', 'repeatactivity']

    def extract_activities(df):
        return dict(zip(df['activity_type'], df['sum_click']))

    def extract_test_dates(df, num_dates=3):
        return pd.Series({'dates': sorted(df['date'].tolist())[:num_dates]})

    def filter_according_to_range(df, using_testing_dates=using_testing_dates,
                                  user_defined_range=user_defined_range):
        if user_defined_range:
            start, end = user_defined_range
        else:
            start = 0
            end = df['dates'][using_testing_dates]

        return df['date'] <= end

    def agg_activities(df, valid_activies=valid_activies):
        df = df.copy()
        df.sort_values('date', inplace=True)
        res = []
        for index in df.index:
            row = []
            for activity in valid_activies:
                try:
                    row.append(df.at[index, 'activities'][activity])
                except KeyError:
                    row.append(0)
            res.append(row)
        return pd.Series({'activities': res})

    def get_activity_regularity(activities, activity_names=valid_activies):
        r = regularity.ActivityRegularity(activity_names, activities)
        return r.get_regularity()

    def unwrap_activity_regularity(df, metrics=['concentration', 'consistency']):
        assert 'regularity' in df.columns
        for metric in metrics:
            df[metric] = df['regularity'].apply(
                lambda x: x[metric] if metric in x else np.nan)
        return df

    # make a copy
    vle = vle.copy()
    studentVle = studentVle.copy()
    assessments = assessments.copy()

    #######################
    # preprocess
    #######################
    identifiers = ['code_module', 'code_presentation', 'id_site']
    studentVle = studentVle.query('date >= 0')
    studentVle = studentVle.merge(vle, on=identifiers, how='inner')

    # aggregate clicks
    identifiers = [
        'code_module', 'code_presentation',
        'id_student', 'date', 'activity_type']
    studentVle = studentVle.groupby(
        identifiers)['sum_click'].agg('sum').reset_index()

    # build features
    identifiers = ['code_module', 'code_presentation', 'id_student', 'date']
    studentVle = studentVle.groupby(
        identifiers).apply(extract_activities).reset_index()
    studentVle.columns = identifiers + ['activities']

    # merge wiht assessments dates
    identifiers = ['code_module', 'code_presentation']
    assessments = assessments.groupby(identifiers).apply(extract_test_dates)
    assessments.reset_index(inplace=True)
    studentVle = studentVle.merge(assessments, on=identifiers, how='left')

    # filter dates
    mask = studentVle.apply(filter_according_to_range, axis=1)
    studentVle = studentVle[mask].copy()

    identifiers = ['code_module', 'code_presentation', 'id_student']
    studentVle = studentVle.groupby(
        identifiers).apply(agg_activities).reset_index()

    studentVle['regularity'] = studentVle['activities'].apply(get_activity_regularity)
    studentVle = unwrap_activity_regularity(studentVle)

    # drop useless columns
    if drop_columns:
        studentVle.drop(['regularity', 'activities'], axis=1, inplace=True)

    return studentVle


def build_procrastination(studentVle, assessments, using_testing_dates=0, user_defined_range=[]):
    # define helper functions
    def extract_timestamps_and_weights(df):
        return pd.Series({
            'timestamps': df['date'].tolist(),
            'weights': df['sum_click'].tolist()})

    def extract_test_dates(df, num_dates=3):
        return pd.Series({'dates': sorted(df['date'].tolist())[:num_dates]})

    def get_procastination(df, fun, 
        using_testing_dates=using_testing_dates, 
        user_defined_range=user_defined_range):
        if user_defined_range:
            start, end = user_defined_range
        elif 0 <= using_testing_dates <= 2:
            start = 0
            end = df['dates'][using_testing_dates]

        ts = df['timestamps']
        ws = df['weights']
        # cut off dates after the end date
        ts = filter(lambda x: x <= end, ts)
        ws = ws[:len(ts)]
        p = procrastination.WeightedMean(fun, ts, ws, end=end, unit='day')
        return p.get_procrastination()

    # make a copy
    studentVle = studentVle.copy()
    assessments = assessments.copy()

    #######################
    # preprocess studentVle
    #######################
    studentVle = studentVle.query('date >= 0')

    # aggregate daily clicks
    identifiers = ['code_module', 'code_presentation', 'id_student', 'date']
    studentVle = studentVle.groupby(
        identifiers)['sum_click'].agg('sum').reset_index()
    studentVle.sort_values(identifiers, inplace=True)

    # apply function
    identifiers = ['code_module', 'code_presentation', 'id_student']
    studentVle = studentVle.groupby(
        identifiers).apply(extract_timestamps_and_weights)
    studentVle.reset_index(inplace=True)

    #######################
    # preprocess assessments
    #######################
    identifiers = ['code_module', 'code_presentation']
    assessments = assessments.groupby(identifiers).apply(extract_test_dates)
    assessments.reset_index(inplace=True)

    # merge
    identifiers = ['code_module', 'code_presentation']
    studentVle = studentVle.merge(assessments, on=identifiers, how='left')

    # build features
    feature_names = ['proc1', 'proc2', 'proc3']
    functions = [lambda x: x,
                 lambda x: 1. / (1 - x),
                 lambda x: math.log(1 + x, 2)]
    for feature_name, f in zip(feature_names, functions):
        studentVle[feature_name] = studentVle.apply(get_procastination, axis=1, fun=f)

    # drop unuseful columns
    studentVle.drop(['timestamps', 'weights', 'dates'], axis=1, inplace=True)

    return studentVle


class preprocessing(object):
    @staticmethod
    def categorical_encoding(feature):
        label_encoder = LabelEncoder()
        onehot_encoder = OneHotEncoder(sparse=False)

        label_encoder.fit(feature)
        feature_labels = label_encoder.transform(feature).reshape(-1, 1)
        matrix = onehot_encoder.fit_transform(feature_labels)

        return pd.DataFrame(
            matrix, index=feature.index, columns=label_encoder.classes_)

    @staticmethod
    def band_average(feature):
        def band_average_helper(x):
            if pd.isnull(x):
                return x
            else:
                band_str = re.findall(r'[\d]+', x)
                band_flt = sum([float(i) for i in band_str]) / len(band_str)
            return band_flt

        ordinal_feature = feature.apply(band_average_helper)
        return ordinal_feature

    @staticmethod
    def preprocessing_studentInfo(studentInfo):
        # make a copy
        studentInfo = studentInfo.copy()

        # create responce
        studentInfo['responce'] = studentInfo['final_result'].apply(
            lambda x: {
                'Distinction': 2,
                'Pass': 2,
                'Fail': 1,
                'Withdrawn': 0
            }[x])
        studentInfo.drop('final_result', axis=1, inplace=True)

        # encode categorical variables
        cat_feature_list = [
            'gender', 'region', 'highest_education', 'disability']
        for col in cat_feature_list:
            feature = studentInfo[col]
            one_hot_coded_feature = preprocessing.categorical_encoding(feature)
            studentInfo.drop(col, axis=1, inplace=True)
            studentInfo[one_hot_coded_feature.columns] = one_hot_coded_feature

        # encode ordinal variables
        ord_feature_list = ['imd_band', 'age_band']
        for col in ord_feature_list:
            feature = studentInfo[col]
            ord_feature = preprocessing.band_average(feature)
            studentInfo.drop(col, axis=1, inplace=True)
            studentInfo[col] = ord_feature

        # filter on responce
        mask = studentInfo['responce'].notnull()

        return studentInfo[mask]

    @staticmethod
    def longest_run(l, digit=0):
        '''input:
                l is a list of floats (0 -> 0, >0 -> 1);
                digit is the digit of interest
            output: the longest run for a digit'''
        longest_run_so_far = 0
        current_run = 0

        for d in l:
            d = math.ceil(d)
            if d == digit:
                current_run += 1
            elif longest_run_so_far > current_run:
                current_run = 0
            else:
                longest_run_so_far = current_run
                current_run = 0

        # handling the case where the longest run is the last run
        return max(longest_run_so_far, current_run)

    @staticmethod
    def preprocessing_studentVle(vle, studentVle, using_testing_dates=0, user_defined_range=[]):
        vle = vle.copy()
        studentVle = studentVle.copy()

        # merge two
        studentVle = studentVle.merge(
            vle,
            on=['code_module', 'code_presentation', 'id_site'],
            how='outer', indicator=True)
        print('the result of merging for studentVle and vle is \n {}'
              .format(studentVle['_merge'].value_counts()))
        studentVle.drop('_merge', axis=1, inplace=True)

        # create identifiers
        studentVle['temperal_seg'] = studentVle['date'].apply(
            lambda x: math.ceil(x / 30.) if pd.notnull(x) else x)
        identifiers = [
            'code_module', 'code_presentation',
            'id_student', 'temperal_seg']

        # effort
        effort = studentVle.groupby(identifiers)[
            'sum_click'].agg('sum').reset_index()

        # marginal distribution on resource
        resource_dist = studentVle.groupby(identifiers + ['activity_type'])[
            'sum_click'].agg('sum').unstack(-1)
        resource_dist.columns = list(resource_dist.columns)
        resource_dist = resource_dist.div(resource_dist.sum(axis=1), axis=0)
        assert np.allclose(resource_dist.sum(axis=1), 1)
        assert resource_dist.sum(axis=1).shape[0] == resource_dist.shape[0]
        resource_dist.reset_index(inplace=True)
        resource_dist.fillna(0, inplace=True)

        # marginal distribution on time:
        studentVle['temperal_seq'] = studentVle['date'] - (
            studentVle['temperal_seg'] - 1) * 30
        temporal_dist = studentVle.groupby(
            identifiers + ['temperal_seq'])['sum_click'].agg('sum').unstack(-1)
        day_seq = ['day ' + str(d) for d in list(temporal_dist.columns)]
        temporal_dist.columns = day_seq
        temporal_dist.fillna(0, inplace=True)
        temporal_dist = temporal_dist.div(temporal_dist.sum(axis=1), axis=0)
        assert np.allclose(temporal_dist.sum(axis=1), 1)
        assert temporal_dist.sum(axis=1).shape[0] == temporal_dist.shape[0]
        temporal_dist.reset_index(inplace=True)

        # build non-linear summary features for temporal_dist
        temporal_dist['peak'] = temporal_dist[day_seq].max(axis=1)
        temporal_dist['variation'] = temporal_dist[day_seq].var(axis=1).apply(math.log)
        temporal_dist['kurtosis'] = temporal_dist[day_seq].kurtosis(axis=1)
        temporal_dist['longest_zeros'] = temporal_dist[day_seq].apply(lambda x: preprocessing.longest_run(x, 0), axis=1)
        temporal_dist['longest_ones'] = temporal_dist[day_seq].apply(lambda x: preprocessing.longest_run(x, 1), axis=1)
        temporal_dist['entropy'] = temporal_dist[day_seq].apply(lambda x: scipy.stats.entropy(x, base=2),axis=1)

        engagement = effort.merge(resource_dist, on=identifiers, how='left')
        engagement = engagement.merge(temporal_dist, on=identifiers, how='left')
        assert engagement[identifiers].duplicated().sum() == 0

        # ignore temperal_seg larger than 8
        return engagement[(engagement['temperal_seg'] < 9)]

    @staticmethod
    def preprocessing_selfregularity_based_features(vle, studentVle, assessments, range):
        proc = build_procrastination(studentVle, assessments)
        time_df = build_timebased_regularity(studentVle, assessments)
        act_df = build_activitybased_regularity(studentVle, vle)



    @staticmethod
    def merge_and_process(studentInfo, engagement):
        studentInfo = studentInfo.copy()
        engagement = engagement.copy()

        df = studentInfo.merge(
            engagement, on=[
                'code_module',
                'code_presentation',
                'id_student'],
            how='left', indicator=True)
        print('''the result of merging for
             studentInfo+assessments
              and engagement is \n {}'''
              .format(df['_merge'].value_counts()))
        df.drop('_merge', axis=1, inplace=True)

        assert df[[
            'code_module', 'code_presentation',
            'id_student', 'temperal_seg']].duplicated().sum() == 0
        # filter on responce
        return df[df['responce'].notnull()]

    @staticmethod
    def fit_transform(name_space):
        studentInfo = preprocessing.preprocessing_studentInfo(
            name_space.studentInfo)
        engagement = preprocessing.preprocessing_studentVle(
            name_space.vle, name_space.studentVle)

        df = preprocessing.merge_and_process(studentInfo, engagement)
        return df
