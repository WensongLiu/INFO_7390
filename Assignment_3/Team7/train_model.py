import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import autosklearn.classification
import sklearn.metrics
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

cls = autosklearn.classification.AutoSklearnClassifier()

def read_txt_file(path, col_header):
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

columns_header = ['LOAN_SEQUENCE_NUMBER','MONTHLY_REPORTING_PERIOD','CURRENT_ACTUAL_UPB','CURRENT_LOAN_DELINQUENCY_STATUS','LOAN_AGE','REMAINING_MONTHS_TO_LEGAL_MATURITY','REPURCHASE_FLAG','MODIFICATION_FLAG','ZERO_BALANCE_CODE','ZERO_BALANCE_EFFECTIVE_DATE','CURRENT_INTEREST_RATE','CURRENT_DEFERRED_UPB','DUE_DATE_OF_LAST_PAID_INSTALLMENT(DDLPI)','MI_RECOVERIES','NET_SALES_PROCEEDS','NON_MI_RECOVERIES','EXPENSES','LEGAL_COSTS','MAINTENANCE_AND_PRESERVATION_COSTS','TAXES_AND_INSURANCE','MISCELLANEOUS_EXPENSES','ACTUAL_LOSS_CALCULATION','MODIFICATION_COST','STEP_MODIFICATION_FLAG','DEFERRED_PAYMENT_MODIFICATION','ESTIMATED_LOAN_TO_VALUE(ELTV)']
df_train = read_txt_file('./Q12015/historical_data1_time_Q12015.txt', columns_header)
x_train, y_train = pre_poccessing(df_train)
df_test = read_txt_file('./Q22015/historical_data1_time_Q22015.txt', columns_header)
x_test, y_test = pre_poccessing(df_test)

cls.fit(x_train,y_train)
save_model(cls, 'automl.sav')

rfclf = RandomForestClassifier(n_estimators = 100)
rfcls.fit(x_train, y_train)
save_model(rfcls, 'rf_cls_trained.sav')

nnclf = MLPClassifier(hidden_layer_sizes = (10,7,5,2))
nncls.fit(x_train, y_train)
save_model(nncls, 'nn_cls_trained.sav')
