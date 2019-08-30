 # Predicting students' failure in online education

 The **End Goal** of this project is to build an early predicting model to identify students' failure in online education. *Failure* is defined as either: drop the course or fail the course.

 Training and Testing data is based on the [Open University Learning Analytics dataset](https://analyse.kmi.open.ac.uk/open_dataset)
 
 To grasp a full picture of this project, please see [abstract](https://github.com/banchisme/open-university/blob/master/Docs/IEEE%20Special%20Issue%20Abstract.pdf)
 
 #### 1. Project Structure
 ``` bash
 open-university/
    data/ # data folder to hold raw data files
    docs/ # main documentation
    notebooks/ # example jupyter notebook code
    university/ # utility package to build the data pipeline
 ```
 #### 2. Quick Start
 
 1. to learn more about this project, go to: 
 [project description](https://github.com/banchisme/open-university/blob/master/docs/IEEE%20Special%20Issue%20Abstract.pdf)
 
 2. to see how to use `university` to build a customized data pipline;
    see `notebooks/example_of_building_data_pipeline.ipynb`.
 
 3. to see some example early prediction model: see `notebooks/example_of_building_a_simple_classification_model.ipynb`
 
 #### 3. Example code for the `university` package
 ```python
from university import data_loader, feature_pool, features

# load raw data
raw = data_loader.get_raw_data()

# split into train/test
train, test = data_loader.train_test_split(raw)

# load pre-defined features
num_features = feature_pool.get_numeric_features_batch(train)
cat_features = feature_pool.get_categorical_features_batch(train)
ord_features = feature_pool.get_ordinal_features_batch(train)

# get response variable
response = feature_pool.get_response(train, targets=[['Pass', 'Distinction'], ['Fail']])

# merge resonse and features
response.update(num_features)
response.update(cat_features)
response.update(ord_features)
train_data = response.merge().data

# seperate into X and y
X = train_data[train_data.columns[1:]]
y = train_data[train_data.columns[0:1]]

```