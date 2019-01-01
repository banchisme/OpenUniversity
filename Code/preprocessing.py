import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer, OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
import math
import scipy
import re


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
    def preprocessing_studentVle(vle, studentVle):
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
