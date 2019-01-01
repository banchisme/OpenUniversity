import pandas as pd

# pk = code_module, code_presentation, id_assessment
assessments = pd.read_csv('../Data/Raw/assessments.csv')
# pk = code_module, code_presentation
courses = pd.read_csv('../Data/Raw/courses.csv')
# pk = id_assessment(this is unique across course sessions), id_student
# pk = code_module, code_presentation, id_student
studentAssessment = pd.read_csv('../Data/Raw/studentAssessment.csv')
studentInfo = pd.read_csv('../Data/Raw/studentInfo.csv')
# pk = code_module, code_presentation, id_student
studentRegistration = pd.read_csv('../Data/Raw/studentRegistration.csv')
# pk = id_site
vle = pd.read_csv('../Data/Raw/vle.csv')
# pk = code_module, code_presentation, id_student, id_site
studentVle = pd.read_csv('../Data/Raw/studentVle.csv')


def train_test_split(df):
    '''this function mimic the train_test_split
    in sklearn.model_selection.train_test_split;
    However, it does not split X and y yet'''

    training_presentation = ['2013B', '2013J', '2014B']
    testing_presentation = ['2014J']
    training_assessments = assessments[assessments['code_presentation'].\
        isin(training_presentation)]['id_assessment'].tolist()
    testing_assessments = assessments[~assessments['code_presentation'].\
        isin(training_presentation)]['id_assessment'].tolist()
    if 'code_presentation' in df.columns:
        train = df[df['code_presentation'].isin(training_presentation)].copy()
        test = df[df['code_presentation'].isin(testing_presentation)].copy()
    elif 'id_assessment' in df.columns:
        train = df[df['id_assessment'].isin(training_assessments)].copy()
        test = df[df['id_assessment'].isin(testing_assessments)].copy()
    else:
        raise NotImplementedError

    return train, test


class NameSpace(object):
    pass


train = NameSpace()
test = NameSpace()
total = NameSpace()
datasets = [
    assessments, courses, studentAssessment,
    studentInfo, studentRegistration, vle, studentVle]

data_names = [
    'assessments', 'courses', 'studentAssessment', 'studentInfo',
    'studentRegistration', 'vle', 'studentVle']
for pt in range(len(datasets)):
    name = data_names[pt]
    df = datasets[pt]
    df_train, df_test = train_test_split(df)
    setattr(train, name, df_train)
    setattr(test, name, df_test)
    setattr(total, name, df)