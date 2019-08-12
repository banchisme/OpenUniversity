import pandas as pd
from . import settings
import os


# low level helper functions
def format_file_name(file_name):
    r""" format file name
    :param file_name (str): raw file name
    :return: formated file name
    """
    res = ''

    for c in file_name:
        if c == '.':
            break   # keep only part before '.'
        elif c.lower() != c:
            res += '_' + c.lower() # change camel separate into underscore separate
        else:
            res += c
    return res


def get_assessment_by_year(years=[]):
    r"""
    :param years (list): year codes
    :return: a list of assessment ids that are in these years
    """

    # assessments table records which year an assessment is given
    assessments = get_raw_data(['assessments.csv'])['assessments']
    mask = assessments['code_presentation'].isin(years)

    res = list(assessments[mask]['id_assessment'].unique())

    return res


# mid level helper functions
def get_raw_data(file_names=[]):
    r"""get raw data
        Argument:
            file_names (list) a list of file names to load
        Return:
            data (dict): {file_name: data as a pandas.DataFrame}
    """
    if len(file_names) == 0:  # load all files if not specified
        file_names = filter(
            lambda f: f.split('.')[-1] == 'csv',
            os.listdir(settings.DATA_DIR))

    data = {}
    for file_name in file_names:
        file_path = os.path.join(settings.DATA_DIR, file_name)
        data_name = format_file_name(file_name)
        data[data_name] = pd.read_csv(file_path)

    return data


# high level functions that user will directly call
def train_test_split(df, train_ratio=None, train_size=None, train_years=['2013B', '2013J', '2014B']):
    r"""
        train test split, one of the train_ratio/train_size/train_years must be not None.

    :param df (pd.DataFrame): dataframe to be splited
    :param train_ratio: train ratio
    :param train_size: train size
    :param train_years: years of data to be included in the training set
    :return: train, test data frame

        Note: only train_years are implemented
    """

    if train_ratio is not None:
        raise NotImplementedError
    if train_size is not None:
        raise NotImplementedError
    if train_years is None:
        raise Exception('the training years cannot be missing')

    if 'code_presentation' in df:  # the tables call year as 'code_presentation'
        col = 'code_presentation'
        val = train_years
    else:
        col = 'id_assessment'
        val = get_assessment_by_year(train_years)

    train_mask = df[col].isin(val)
    train_df = df[train_mask].copy()
    test_df = df[~ train_mask].copy()

    return train_df, test_df







class NameSpace(object):
    pass


train = NameSpace()
test = NameSpace()
total = NameSpace()
# datasets = [
#     assessments, courses, studentAssessment,
#     studentInfo, studentRegistration, vle, studentVle]
#
# data_names = [
#     'assessments', 'courses', 'studentAssessment', 'studentInfo',
#     'studentRegistration', 'vle', 'studentVle']
# for pt in range(len(datasets)):
#     name = data_names[pt]
#     df = datasets[pt]
#     df_train, df_test = train_test_split(df)
#     setattr(train, name, df_train)
#     setattr(test, name, df_test)
#     setattr(total, name, df)


