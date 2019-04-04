import pandas as pd
from sklearn.feature_extraction import DictVectorizer

import sklearn.metrics
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import random
import boto3
import os
from botocore.client import Config
import numpy as np
from tpot import TPOTClassifier
import h2o
from h2o.estimators.deepleearning import H2ODeepLearningEstimator


year = sys.argv[1]
quater = sys.argv[2]
bucket_name = sys.argv[3]
ACCESS_KEY_ID = sys.argv[4]
ACCESS_SECRET_KEY = sys.argv[5]
print('Please choose a model: \n1 -> {}\n2 -> {}\n3 -> {}\n4 -> {}\n5 ->{}\n6 ->{}'.format('LogisticRegression by Sklearn','MLPClassifier by Sklearn','RandomForestClassifier by Sklearn','AutoSklearnClassifier by AutoSklearn','TPOTClassifier by TPOT','H2OMultinomialModel by H2O.ai'))
model_chooser = sys.argv[6]

s3 = boto3.client('s3', aws_access_key_id = ACCESS_KEY_ID, 
                   aws_secret_access_key = ACCESS_SECRET_KEY,
                   config = Config(signature_version = 's3v4'))

def create_bucket():
    response = s3.list_buckets()
    buckets = [bucket['Name'] for bucket in response['Buckets']]
    #print("Bucket List: %s" % buckets)
    for i in range(len(buckets)):
        if buckets[i] == bucket_name:
            break
        if i == len(buckets)-1:
            print(bucket_name)
            s3.create_bucket(Bucket = bucket_name)

def upload_to_s3(folderpath,bucket_name,filename):
    s3.upload_file(folderpath,bucket_name,filename)

columns_header = ['LOAN_SEQUENCE_NUMBER','MONTHLY_REPORTING_PERIOD','CURRENT_ACTUAL_UPB','CURRENT_LOAN_DELINQUENCY_STATUS','LOAN_AGE','REMAINING_MONTHS_TO_LEGAL_MATURITY','REPURCHASE_FLAG','MODIFICATION_FLAG','ZERO_BALANCE_CODE','ZERO_BALANCE_EFFECTIVE_DATE','CURRENT_INTEREST_RATE','CURRENT_DEFERRED_UPB','DUE_DATE_OF_LAST_PAID_INSTALLMENT(DDLPI)','MI_RECOVERIES','NET_SALES_PROCEEDS','NON_MI_RECOVERIES','EXPENSES','LEGAL_COSTS','MAINTENANCE_AND_PRESERVATION_COSTS','TAXES_AND_INSURANCE','MISCELLANEOUS_EXPENSES','ACTUAL_LOSS_CALCULATION','MODIFICATION_COST','STEP_MODIFICATION_FLAG','DEFERRED_PAYMENT_MODIFICATION','ESTIMATED_LOAN_TO_VALUE(ELTV)']

def path_compiler():
    path = './Q{}{}/historical_data1_time_Q{}{}.txt'.format(quater,year,quater,year)
    if not os.path.isfile(path):
        raise Exception("File does not exists!")
    else:
        return path;

def read_txt_file(col_header):
    path = path_compiler()
    df = pd.read_table(path, delimiter='|', header = None, names = col_header)
    return df

def dropCol(dataset, cutoff = 0.3):
    for col_name in columns_header:
        n = len(dataset)
        value_count = dataset[col_name].count()
        if(value_count/n) < cutoff:
            dataset.drop(col_name, axis = 1, inplace = True)
    return dataset
  
def toDatetime(dataset):
    dataset["MONTHLY_REPORTING_PERIOD"] = pd.to_datetime(dataset["MONTHLY_REPORTING_PERIOD"], format = "%Y%m")
    dataset['Year'] = pd.DatetimeIndex(dataset["MONTHLY_REPORTING_PERIOD"]).year
    dataset['Month'] = pd.DatetimeIndex(dataset["MONTHLY_REPORTING_PERIOD"]).month
    dataset.drop(["MONTHLY_REPORTING_PERIOD"], axis = 1, inplace = True)
    return dataset

def categorical_to_numerical(dataset):
    categorical_columns = dataset.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        dataset[col] = dataset[col].astype('category')
    vec = DictVectorizer(sparse=False, dtype=int)
    dc = dataset.to_dict('records')
    result = vec.fit_transform(dc)
    return result

def divide_x_y(dataframe):
    Y = dataframe['CURRENT_LOAN_DELINQUENCY_STATUS']
    dataframe.drop(['CURRENT_LOAN_DELINQUENCY_STATUS'], axis = 1, inplace = True)
    df_Y = pd.DataFrame(Y)
    return dataframe, df_Y

def pre_poccessing(df):
    df = dropCol(df)
    df = toDatetime(df)
    x,y = divide_x_y(df)
    x.drop(['LOAN_SEQUENCE_NUMBER'], axis = 1, inplace = True)
    y = y.replace(to_replace=r'.*R.*', value = '-1', regex = True)
    y = y['CURRENT_LOAN_DELINQUENCY_STATUS'].values.astype('int32')
    x = categorical_to_numerical(x)
    return x, y

def save_model(model, filename):
    pickle.dump(model, open(filename, 'wb'))

def build_fit_model(x_train, y_train):
    if model_choice == '6':
        clf = model_chooser()
        clf.train(x=x_train, y=y_train, train_frame=fr)
    else:
        clf = model_chooser()
        clf.fit(x_train, y_train)
    return clf

def model_chooser():
    if model_choice == '1':
        return LogisticRegression(random_state = 0, solver = 'lbfgs', multi_class = 'multinomial')
    elif model_choice == '2':
        return MLPClassifier(hidden_layer_sizes = (10,7,5,2))
    elif model_choice == '3':
        return RandomForestClassifier(n_estimators = 100)
    elif model_choice == '4':
        import autosklearn.classification
        return autosklearn.classification.AutoSklearnClassifier()
    elif model_choice == '5':
        return TPOTClassifier(generation=5, population_size=20, verbosity=2)
    else:
        h2o.init()
        return H2ODeepLearningEstimator()

def main():
    print("Start reading data and prepoccessing.")
    pd_train = read_txt_file(columns_header)
    x_train, y_train = pre_poccessing(pd_train)
    print("Start building and training model.")
    clf = build_fit_model(x_train, y_train)
    print("training finished, ready to save.")
    model_id = random.randint(10000,99999)
    model_filename = 'trained_model_{}_().sav'.format(model_chooser,str(model_id))
    save_model(clf, model_filename)
    print('Model saved locally, ready to upload to AWS S3 bucket')
    print('Start creating s3 bucket.')
    create_bucket()
    print('s3 bucket created, start uploading')
    upload_to_s3('./'+ model_filename, bucket_name, model_filename)
    print('Congrats! Model has uploaded to AWS s3 bucket, bucketname: {}'.format(bucket_name))
    
if __name__ == '__main__':
    main()