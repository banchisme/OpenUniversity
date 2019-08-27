import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer, OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import roc_auc_score, f1_score, make_scorer
from sklearn.svm import SVC
from scipy import stats
import math
from tqdm import tqdm_notebook
import re


identifiers = ['code_module', 'code_presentation', 'id_student', 'temperal_seg']
responce = ['responce']
time_invariant_features = ['num_of_prev_attempts', 'studied_credits', 'F', 'M',
       'East Anglian Region', 'East Midlands Region', 'Ireland',
       'London Region', 'North Region', 'North Western Region', 'Scotland',
       'South East Region', 'South Region', 'South West Region', 'Wales',
       'West Midlands Region', 'Yorkshire Region', 'A Level or Equivalent',
       'HE Qualification', 'Lower Than A Level', 'No Formal quals',
       'Post Graduate Qualification', 'N', 'Y', 'imd_band', 'age_band', 'date_registration']
assessment_features = ['date_submitted','is_banked', 'score', 'weighted_score', 'date_in_advance']

effort_features = ['sum_click']
resource_features = ['dataplus', 'dualpane', 'externalquiz', 'forumng', 'glossary',
       'homepage', 'htmlactivity', 'oucollaborate', 'oucontent',
       'ouelluminate', 'ouwiki', 'page', 'questionnaire', 'quiz', 'resource', 'sharedsubpage', 'subpage', 'url']
temporal_features = ['day 1.0', 'day 2.0', 'day 3.0', 'day 4.0', 'day 5.0', 'day 6.0', 'day 7.0', 'day 8.0',
       'day 9.0', 'day 10.0', 'day 11.0', 'day 12.0', 'day 13.0', 'day 14.0', 'day 15.0',
       'day 16.0', 'day 17.0', 'day 18.0', 'day 19.0', 'day 20.0', 'day 21.0', 'day 22.0',
       'day 23.0', 'day 24.0', 'day 25.0', 'day 26.0', 'day 27.0', 'day 28.0', 'day 29.0',
       'day 30.0']
temporal_summary_features = ['peak', 'variation', 'kurtosis', 'longest_zeros',
       'longest_ones']


def train_test_split(df):
    '''this function mimic the train_test_split in sklearn.model_selection.train_test_split; However, it does not split X and y yet'''
    training_presentation = ['2013B', '2013J', '2014B']
    testing_presentation = ['2014J']
    training_assessments = assessments[assessments['code_presentation'].\
        isin(training_presentation)]['id_assessment'].tolist()
    testing_assessments = assessments[~assessments['code_presentation'].\
        isin(training_presentation)]['id_assessment'].tolist()
    if 'code_presentation' in df.columns:
        train = df[df['code_presentation'].isin(training_presentation)]
        test = df[df['code_presentation'].isin(testing_presentation)]
    elif 'id_assessment' in df.columns:
        train = df[df['id_assessment'].isin(training_assessments)]
        test = df[df['id_assessment'].isin(testing_assessments)]
    else:
        raise NotImplementedError
    
    return train, test 

class preprocessing(object):
    @staticmethod
    def categorical_encoding(feature):
        label_encoder = LabelEncoder()
        onehot_encoder = OneHotEncoder(sparse=False)
        
        label_encoder.fit(feature)
        feature_labels = label_encoder.transform(feature).reshape(-1, 1)
        matrix = onehot_encoder.fit_transform(feature_labels)
        
        return pd.DataFrame(matrix, index=feature.index, columns=label_encoder.classes_)

    @staticmethod
    def band_average(feature):
        def band_average_helper(x):
            if pd.isnull(x):
                return x
            else:
                band_str = re.findall('[\d]+', x)
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
            lambda x: {'Distinction': 0,'Pass': 0, 'Fail': 1, 'Withdrawn': np.nan}[x])
        studentInfo.drop('final_result', axis=1, inplace=True)
        
        # encode categorical variables
        cat_feature_list = ['gender', 'region', 'highest_education', 'disability']
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
        studentVle = studentVle.merge(vle, on=['code_module', 'code_presentation', 'id_site'], how='outer', indicator=True)
        print('the result of merging for studentVle and vle is \n {}'
              .format(studentVle['_merge'].value_counts()))
        studentVle.drop('_merge', axis=1, inplace=True)
        
        # create identifiers
        studentVle['temperal_seg'] = studentVle['date'].apply(lambda x: math.ceil(x / 30.) if pd.notnull(x) else x)
        identifiers = ['code_module', 'code_presentation', 'id_student', 'temperal_seg']
        
        # effort
        effort = studentVle.groupby(identifiers)['sum_click'].agg('sum').reset_index()
        
        # marginal distribution on resource
        resource_dist = studentVle.groupby(identifiers + ['activity_type'])['sum_click'].agg('sum').unstack(-1)
        resource_dist.columns = list(resource_dist.columns)
        resource_dist = resource_dist.div(resource_dist.sum(axis=1), axis=0)
        assert np.allclose(resource_dist.sum(axis=1), 1)
        assert resource_dist.sum(axis=1).shape[0] == resource_dist.shape[0]
        resource_dist.reset_index(inplace=True)
        resource_dist.fillna(0, inplace=True)
        
        # marginal distribution on time
        studentVle['temperal_seq'] = studentVle['date'] - (studentVle['temperal_seg'] - 1) * 30
        temporal_dist = studentVle.groupby(identifiers + ['temperal_seq'])['sum_click'].agg('sum').unstack(-1)
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
        
        engagement = effort.merge(resource_dist, on=identifiers, how='left')
        engagement = engagement.merge(temporal_dist, on=identifiers, how='left')
        assert engagement[identifiers].duplicated().sum() == 0
        
        # ignore temperal_seg larger than 8
        return engagement[(engagement['temperal_seg'] < 9)]


    @staticmethod
    def merge_and_process(studentInfo, engagement):
        studentInfo = studentInfo.copy()
        engagement = engagement.copy()
        
        df  = studentInfo.merge(engagement, on=['code_module', 'code_presentation', 'id_student'], how='left', indicator=True)
        print('the result of merging for studentInfo+assessments and engagement is \n {}'
              .format(df['_merge'].value_counts()))
        df.drop('_merge', axis=1, inplace=True)
        
        assert df[['code_module', 'code_presentation', 'id_student', 'temperal_seg']].duplicated().sum() == 0
        # filter on responce
        return df[df['responce'].notnull()]

    @staticmethod
    def fit_transform(name_space):
        studentInfo = preprocessing.preprocessing_studentInfo(name_space.studentInfo)
        engagement = preprocessing.preprocessing_studentVle(name_space.vle, name_space.studentVle)
        
        df = preprocessing.merge_and_process(studentInfo, engagement)
        return df


class TemporalSegments(object):
    def __init__(self, data, num_segments=8):
        self.temporal_segments = {}
        # create and append temporal_segments
        for i in range(0, num_segments + 1):
            self.temporal_segments[i] = self.create_temporal_segment(data, i)
            if i > 1:
                self.temporal_segments[i] = self.append_temporal_segment(self.temporal_segments[i - 1],self.temporal_segments[i])
        # seperate X and y, and drop identifiers
        for i in self.temporal_segments:
            X_y = self.temporal_segments[i]
            y = X_y['responce']
            X = X_y.drop(['code_module', 'code_presentation', 'id_student', 'temperal_seg', 'responce'], axis=1)
            self.temporal_segments[i] = (X, y)
            
    def create_temporal_segment(self, data, seg):
        # filtering
        data = data[data['temperal_seg'] == seg]
        # rename features
        columns = []
        for col in data.columns:
            if col not in identifiers + responce + time_invariant_features:
                columns.append('t' + str(seg) + '_' + col)
            else:
                columns.append(col)   
        data.columns = columns  
        return data
    
    def append_temporal_segment(self, t1, t2):
        '''time varying features in t1 are appended into t2'''
        merge_conditions = ['code_module', 'code_presentation', 'id_student']
        t1_useful_features = []
        for col in t1.columns:
            if col not in identifiers + time_invariant_features:
                t1_useful_features.append(col)
    
        t1_useful = t1[merge_conditions + t1_useful_features]
        t2 = t2.merge(t1_useful, on=merge_conditions, how='outer', indicator=True)
        
        # join responce
        t2['responce'] = t2[['responce_x', 'responce_y', '_merge']].apply(
            lambda x: x['responce_y'] if x['_merge'] == 'right_only' else x['responce_x'], axis=1)
        t2.drop(['responce_x', 'responce_y', '_merge'], axis=1, inplace=True)
        
        assert t2[merge_conditions].duplicated().sum()==0
        return t2
    
    def get_temporal_segment(self, t):
        assert 0 <= t <=8
        return self.temporal_segments[t]


def my_pipeline(t, num_features, max_num_features=100):
    '''input: t is the temporal segment number, max_num_features is used as a cap for feature selection. 
    output: a sklearn.pipeline.Pipeline'''
    
    half_num_features = math.floor(num_features / 2)
    final_num_features = min(half_num_features, max_num_features)
    f_select = FeatureUnion([('pca', PCA(final_num_features)), 
                            ('bestk', SelectKBest(k=final_num_features))])
    pipe = Pipeline([('imput',Imputer()), ('std', StandardScaler()), ('f_select', f_select)])
    
    return pipe

def temporal_feature_name(feature_cat, seg):
    if feature_cat == 'effort_features':
        old_features = effort_features
    elif feature_cat == 'resource_features':
        old_features = resource_features
    elif feature_cat == 'temporal_features':
        old_features = temporal_features
    elif feature_cat == 'temporal_summary_features':
        old_features = temporal_summary_features

    new_features = []
    for t in range(1, seg + 1):
        for f in old_features:
            new_features.append('t' + str(t) + '_' + f)

    return new_features
