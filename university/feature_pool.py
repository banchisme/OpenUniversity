r"""
    this module is the place where all the features are defined
"""

import pandas as pd
from university.features import Feature, FeatureDict
from university import preprocessing
from clickfeatures import regularity
from collections import defaultdict


# low level helper functions
def get_test_schedule(assessments, max_n=3):
    r"""get the testing dates for assessments
        Argument:
            assessments (pd.DataFrame): the assessment data
            max_n (int): the number of test dates to include
        Return:
            assessments as a dict {('code_module', 'code_presentation'): [test_dates]}
    """

    def extract_test_dates(df, max_n=max_n):
        return pd.Series({'dates': sorted(df['date'].tolist())[:max_n]})

    assessments = assessments.copy()
    schedule = assessments.groupby(['code_module', 'code_presentation']).apply(extract_test_dates)

    # wrap schedule into a python dict
    return dict(zip(schedule.index, schedule['dates']))


def get_aggregate_clickstream(raw, by=['date'], agg_fun='sum'):
    r"""aggregrate student clickstream data
        Argument:
            raw (pd.DataFrame): the raw dataset container
            by (list): aggreation method
            agg_fun (str): valid aggreation function accepted by pd.grouped.agg
        Return:
            aggregated clickstream data as a pd.DataFrame
    """

    # get needed data
    student_vle = raw['student_vle'].copy()
    vle = raw['vle'].copy()
    student_vle = student_vle.merge(
        vle, on=['code_module', 'code_presentation', 'id_site'],
        how='left')

    # aggregate
    identifiers = ['code_module', 'code_presentation', 'id_student'] + by
    student_vle = student_vle.groupby(identifiers)['sum_click'].agg(agg_fun)

    return student_vle.reset_index()


def get_clickstream_timestamps(raw, start, end):
    r"""get students clickstream timestamps: the date when a click happens
        Argument:
            raw (dict) raw data container
        Return:
            student timestamps as a pandas datafram:
            index: (code_module, code_presentation, student_id)
            columns: [timestamps], [intensity]}
    """

    def apply_fun(df):
        df = df.query('date >= {} & date <= {}'.format(start, end))
        df = df.sort_values('date')

        return pd.Series({
            'timestamps': df['date'].tolist(),
            'intensity': df['sum_click'].tolist()})

    clickstream = get_aggregate_clickstream(raw, by=['date'])

    # extract timestamps and intensity
    ts = clickstream.groupby(
        ['code_module', 'code_presentation', 'id_student']
        ).apply(apply_fun)

    # wrap in to dict
    return ts


def get_dataframe_index(data_frame_name):
    r"""return a data frame's index columns as a list
    Argument:
        data_frame_name: name of data frame
    Return:
        [index columns]
    """
    index_table = {
        'student_assessment': ['id_assessment', 'id_student'],
        'student_info': ['code_module', 'code_presentation', 'id_student'],
        'courses': ['code_module', 'code_presentation'],
        'vle': ['id_site', 'code_module', 'code_presentation'],
        'student_registration': ['code_module', 'code_presentation', 'id_student'],
        'assessments': ['code_module', 'code_presentation', 'id_assessment']
    }

    return index_table[data_frame_name]


def get_ordinal_data_orders(data_frame_name, feature_name):
    r""" return the orders for ordinal data by dataframe name and feature name
    :param data_frame_name (str): data frame name
    :param feature_name (str): column name
    :return: order as a list
    """

    order_table = {
        ('student_info', 'highest_education'):
            ['No Formal quals', 'Lower Than A Level', 'A Level or Equivalent',
             'HE Qualification', 'Post Graduate Qualification'],
        ('student_info', 'imd_band'):
            ['0-10%', '10-20', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%'],
        ('student_info', 'age_band'):
            ['0-35', '35-55', '55<='],
    }

    return order_table[(data_frame_name, feature_name)]


# features
def get_time_regularity(raw, start, end, metrics=['pwd', 'ws1', 'ws2', 'ws3', 'fwd']):
    r""" get students clickstream data time regularity
    :param raw (dict): raw data container
    :param start: clickstream start date
    :param end: clickstream end date
    :return: feature container that contain time regularity features
    """

    # build clickstream timestamps
    clickstream_timestamps = get_clickstream_timestamps(raw, start, end)

    # build time regularity
    data = []

    for _, (timestamps, intensity) in clickstream_timestamps.iterrows():
        if len(timestamps) > 0:
            reg = regularity.TimeRegularity(ts=timestamps, weights=intensity, unit='day')
            data.append(reg.get_regularity())
        else:
            data.append({})

    # wrap time regularity into features
    data = pd.DataFrame(data, index=clickstream_timestamps.index)[metrics]
    data.columns = ['regularity_indicator_' + _ for _ in data.columns]

    feature_container = FeatureDict()

    for col_name in data.columns:
        feature_container[col_name] = Feature(col_name, data[[col_name]])

    return feature_container


def get_procrastination():
    r"""get students """
    raise NotImplementedError


def get_effort(raw, start, end):
    r""" get students clickstream data based effort
    :param raw (dict): raw data container
    :param start: clickstream start date
    :param end: clickstream end date
    :return: feature container that contain effort feature
    """
    clickstream_timestamps = get_clickstream_timestamps(raw, start, end)

    # build effort
    data = []
    feature_name = 'effort'

    for _, (_, intensity) in clickstream_timestamps.iterrows():
        data.append({feature_name: sum(intensity)})

    # wrap effort into features
    data = pd.DataFrame(data, index=clickstream_timestamps.index)
    feature_container = FeatureDict()
    feature_container[feature_name] = Feature(feature_name, data)

    return feature_container


# feature batches
def get_feature_batch(raw, feature_list, preprocessors):
    r"""
    extract numeric features from source data
    :param raw (dict): raw data container
    :param feature_list: [(data_frame_name, feature_name)]
    :param preprocessors (list) each element is an instance of the NumericData, CategoricalData, OrdinalData)
            if len(feature_list) > len(preprocessors), then the last preprocessor is used for the rest of the features
    :return:
        a container of features (features.FeatureDict)
    """

    # book a container
    feature_container = FeatureDict()

    for i, (data_frame_name, feature_name) in enumerate(feature_list):
        # extract feature data from raw
        extractor = preprocessing.ColumnExtractor(
            raw[data_frame_name],
            index_col=get_dataframe_index(data_frame_name))
        feature_data = extractor.extract(feature_name)

        # preprocess the feature data
        feature_preprocessor = preprocessors[min(len(preprocessors) - 1, i)]
        feature_data = feature_preprocessor.fit_transform(feature_data)

        # pack in the feature container
        feature = Feature(feature_name, feature_data)
        feature_container[feature_name] = feature

    return feature_container


def get_numeric_features_batch(raw):
    r"""extract predefined numerical features from raw data
    :param raw (dict): raw data container
    :return: a container of numeric features
    """
    # define feature list here
    predefined_feature_list = [
        ('student_info', 'num_of_prev_attempts'),
        ('student_info', 'studied_credits'),
        ('student_registration', 'date_registration')]

    preprocessors = [preprocessing.NumericData()]

    return get_feature_batch(raw, predefined_feature_list, preprocessors)


def get_categorical_features_batch(raw):
    r"""extracct predefined categorical features from raw data
        Argument:
            raw (dict): raw data container
        Return:
            a container of categorical features
    """

    # define feature list here
    predefined_features = [
        ('student_info', 'gender'),
        ('student_info', 'region'),
        ('student_info', 'disability')]

    preprocessors = [preprocessing.CategoricalData()]

    return get_feature_batch(raw, predefined_features, preprocessors)


def get_ordinal_features_batch(raw):
    r""" extract predefined ordinal features from raw data
    :param raw: (dict): raw data container
    :return: a container of ordinal features
    """
    # define features here
    predefined_features = [
        ('student_info', 'highest_education'),
        ('student_info', 'imd_band'),
        ('student_info', 'age_band')]

    orders = [get_ordinal_data_orders(*args) for args in predefined_features]
    preprocessors = [preprocessing.OrdinalData(order) for order in orders]
    return get_feature_batch(raw, predefined_features, preprocessors)


# response
def get_response(raw, targets=[], response_name='responce', drop_other=True):
    r"""get the responce from the raw data
        Argument:
            raw (dict): dictionary that contains the raw data files (each as a pd.DataFrame)
            targets (list of list):
                sub lists  are target labels, must be a subset of ['Pass', 'Withdrawn', 'Fail', 'Distinction']
                when need to group a subset labels together, e.g., 'Pass' and 'Distinction' together,
                use targets = [['Pass'], ['Distinction']] if only interested to classify between 'Pass' and 'Distinction'
            responce_name (str): the name you want to give to the responce, optional.
            drop_other (bool): if True, drop rows whose response value not in targets
        Return:
            a feature container that contains only the response feature
    """

    # create target encoder
    encoder = defaultdict(lambda: None)
    for i, t in enumerate(targets):
        encoder.update(dict(zip(t, [i] * len(t))))

    # extract target data
    data_frame_name = 'student_info'
    extractor = preprocessing.ColumnExtractor(
            raw[data_frame_name],
            index_col=get_dataframe_index(data_frame_name))
    target_data = extractor.extract('final_result')

    # encode target data
    target_data.columns = [response_name]
    target_data[response_name] = target_data[response_name].map(encoder)

    # drop rows if neccessary
    if drop_other is True:
        target_data.dropna(inplace=True)

    # wrap target_data into a feature_container, this makes merging with other features easier
    target_feature = Feature(response_name, target_data)
    feature_container = FeatureDict()
    feature_container[target_feature.name] = target_feature

    return feature_container
