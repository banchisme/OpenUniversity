r"""
    this module is the place where all the features are defined
"""

import pandas as pd
import bisect
from university.features import Feature, FeatureDict
from university import preprocessing
from clickfeatures import regularity


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
            student timestamps as a dict:
            {(code_module, code_presentation, student_id): [timestamps], [intensity]}
    """

    def apply_fun(df):
        df = df.query('date >= {} & date <= {}'.format(start, end))
        df = df.sort_values('date')

        return (list(df['date'].values),
                list(df['sum_click'].values))

    clickstream = get_aggregate_clickstream(raw, by=['date'])

    # extract timestamps and intensity
    ts = clickstream.groupby(
        ['code_module', 'code_presentation', 'id_student']
        ).apply(apply_fun)

    # wrap in to dict
    return dict(zip(ts.index, ts.values))


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
    index = []
    data = []

    for key, (timestamps, intensity) in clickstream_timestamps.items():
        index.append(key)
        if len(timestamps) > 0:
            reg = regularity.TimeRegularity(ts=timestamps, weights=intensity, unit='day')
            data.append(reg.get_regularity())
        else:
            data.append({})

    # wrap time regularity into features
    data = pd.DataFrame(data, index=index)[metrics]
    data.columns = ['regularity_indicator_' + _ for _ in data.columns]

    feature_container = FeatureDict()

    for col_name in data.columns:
        feature_container[col_name] = Feature(col_name, data[[col_name]])

    return feature_container


def get_procrastination():
    pass


def get_effort(raw, start, end):
    r""" get students clickstream data based effort
    :param raw (dict): raw data container
    :param start: clickstream start date
    :param end: clickstream end date
    :return: feature container that contain effort feature
    """
    clickstream_timestamps = get_clickstream_timestamps(raw, start, end)

    # build effort
    index = []
    data = []
    feature_name = 'effort'

    for key, (_, intensity) in clickstream_timestamps.items():
        index.append(key)
        data.append({feature_name: sum(intensity)})

    # wrap effort into features
    data = pd.DataFrame(data, index=index)
    feature_container = FeatureDict()
    feature_container[feature_name] = Feature(feature_name, data)

    return feature_container