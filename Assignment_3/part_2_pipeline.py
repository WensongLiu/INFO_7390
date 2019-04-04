import re
from requests_html import HTMLSession
import http.cookiejar
import urllib.request as request
import ssl
import time
import zipfile
import os

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler  
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing

from sklearn.model_selection import GridSearchCV
from itertools import chain, combinations
from sklearn.model_selection import cross_val_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


##########Downlaod data
# build a session
session = HTMLSession()
session.cookies = http.cookiejar.LWPCookieJar('cookie')

headers = {'Host':'freddiemac.embs.com',
           'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36',
           'Referer':'https://freddiemac.embs.com/FLoan/secure/login.php?pagename=download'}
def get_cookie():
	try:
		session.cookies.load(ignore_discard = True)
	except IOError:
		print('Cannot load cookie!')
		
def login(un, pwd):
	"""
	entering username and password
	"""
	auth_url = 'https://freddiemac.embs.com/FLoan/secure/auth.php'
	auth_data = {'pagename':'download','username': un,'password': pwd}
	result = session.post(auth_url, data = auth_data, headers = headers, allow_redirects = False)
	term_data = {'accept': 'Yes','action': 'acceptTandC','acceptSubmit': 'Continue'}
	download_url = 'https://freddiemac.embs.com/FLoan/Data/download.php'
	response = session.post(download_url, data = term_data, headers = headers, allow_redirects = False)
	return response

def isLogin():
	url = 'https://freddiemac.embs.com/FLoan/Data/download.php'
	login_code = session.get(url, headers = headers, allow_redirects = False).status_code
	if login_code == 200:
		return True
	else:
		return False
		
def get_download_link(year,quater):
	try:
		year = int(year)
	except ValueError:
		print('year invalid!')
	try:
		quater = int(quater)
	except ValueError:
		print('quater invalid!')
	if year < 1999 or year > 2017:
		raise ValueError('year invalid. year should between 1999 and 2017')
	if quater < 1 or quater > 4:
		raise ValueError('quater invalid. quater should between 1 and 4')
	url = 'https://freddiemac.embs.com/FLoan/Data/download.php'
	response = session.get(url, headers = headers, allow_redirects = False)
	links = response.html.links
	domain = 'https://freddiemac.embs.com/FLoan/Data/'
	yq = 'Q' + str(quater) + str(year)
	pattern = re.compile('.+' + yq + '.+')
	url = ''
	for link in links:
		if pattern.fullmatch(link) != None:
			url = link
			break
	if url == '':
		print('no such file!')
		return None
	download_link = domain + url
	return download_link
	
def download_from_url(url,filename):
	response = session.get(url, headers = headers, allow_redirects = True)
	with open(filename,'wb') as f:
		f.write(response.content)

def get_test_date(year_train, quater_train):
	if (quater_train == 4):
		year_test = year_train + 1
		quater_test = 1
	else:
		year_test = year_train
		quater_test = quater_train + 1
	return year_test, quater_test

def un_zip(year, quater):
	folder_path = 'Q{}{}'.format(quater,year)
	filename = 'Q{}{}.zip'.format(quater,year)
	zip_file = zipfile.ZipFile(filename)
	if os.path.isdir(folder_path):
		pass
	else:
		os.mkdir(folder_path)
	for name in zip_file.namelist():
		zip_file.extract(name,folder_path + '/')
	zip_file.close()

def dropCol(dataset, cutoff = 0.3):
    columns_header = ["CREDIT_SCORE","FIRST_PAYMENT_DATE","FIRST_TIME_HOMEBUYER_FLAG","MATURITY_DATE","MSA","MORTGAGE_INSURANCE_PERCENTAGE","NUMBER_OF_UNITS","OCCUPANCY_STATUS","ORIGINAL_COMBINED_LOAN_TO_VALUE","ORIGINAL_DEBT_TO_INCOME_RATIO","ORIGINAL_UPB","ORIGINAL_LOAN_TO_VALUE","CHANNEL","PREPAYMENT_PENALTY_MORTGAGE_FLAG","PRODUCT_TYPE","PROPERTY_STATE","PROPERTY_TYPE","POSTAL_CODE","LOAN_SEQUENCE_NUMBER","LOAN_PURPOSE","ORIGINAL_LOAN_TERM","NUMBER_OF_BORROWERS","SELLER_NAME","SERVICER_NAME","Pre_HARP_LOAN_SEQUENCE_NUMBER"]
    for col_name in columns_header:
        n = len(dataset)
        value_count = dataset[col_name].count()
        if(value_count/n) < cutoff:
            dataset.drop(col_name, axis = 1, inplace = True)
    return dataset

def toDatetime(dataset):
    dataset["FIRST_PAYMENT_DATE"] = pd.to_datetime(dataset["FIRST_PAYMENT_DATE"], format = "%Y%m")
    dataset["MATURITY_DATE"] = pd.to_datetime(dataset["MATURITY_DATE"], format = "%Y%m")
    dataset['FIRST_PAYMENT_Year'] = pd.DatetimeIndex(dataset["FIRST_PAYMENT_DATE"]).year
    dataset['FIRST_PAYMENT_Month'] = pd.DatetimeIndex(dataset["FIRST_PAYMENT_DATE"]).month
    dataset['MATURITY_DATE_Year'] = pd.DatetimeIndex(dataset["MATURITY_DATE"]).year
    dataset['MATURITY_DATE_Month'] = pd.DatetimeIndex(dataset["MATURITY_DATE"]).month
    dataset.drop(["FIRST_PAYMENT_DATE","MATURITY_DATE"], axis = 1, inplace = True)
    return dataset

def replace_NaN(dataset):
    dataset["CREDIT_SCORE"].replace(9999, float('nan'), inplace = True)
    dataset["MORTGAGE_INSURANCE_PERCENTAGE"].replace(999, float('nan'), inplace = True)
    dataset["NUMBER_OF_UNITS"].replace(99, float('nan'), inplace = True)
    dataset["ORIGINAL_COMBINED_LOAN_TO_VALUE"].replace(999, float('nan'), inplace = True)
    dataset["ORIGINAL_DEBT_TO_INCOME_RATIO"].replace(999, float('nan'), inplace = True)
    dataset["ORIGINAL_LOAN_TO_VALUE"].replace(999, float('nan'), inplace = True)
    dataset["NUMBER_OF_BORROWERS"].replace(99, float('nan'), inplace = True)
    return dataset

def replace_O(dataset):
    dataset["OCCUPANCY_STATUS"].replace("9", "O", inplace = True)
    dataset["CHANNEL"].replace("9", "O", inplace = True)
    dataset["PROPERTY_TYPE"].replace("99", "O", inplace = True)
    dataset["LOAN_PURPOSE"].replace("9", "O", inplace = True)
    return dataset

def flag_toValue(dataset):
    dataset["FIRST_TIME_HOMEBUYER_FLAG"].replace("Y", 1, inplace = True)
    dataset["FIRST_TIME_HOMEBUYER_FLAG"].replace("N", 0, inplace = True)
    dataset["PREPAYMENT_PENALTY_MORTGAGE_FLAG"].replace("Y", 1, inplace = True)
    dataset["PREPAYMENT_PENALTY_MORTGAGE_FLAG"].replace("N", 0, inplace = True)
    
    ppmList = dataset["PREPAYMENT_PENALTY_MORTGAGE_FLAG"].tolist()
    p = ppmList.count(1)
    q = ppmList.count(0)
    if(p>q):
        dataset["PREPAYMENT_PENALTY_MORTGAGE_FLAG"].fillna(1, inplace = True)
    else:
        dataset["PREPAYMENT_PENALTY_MORTGAGE_FLAG"].fillna(0, inplace = True)
    
    firstList = dataset["FIRST_TIME_HOMEBUYER_FLAG"].tolist()
    m = ppmList.count(1)
    n = ppmList.count(0)
    if(m>n):
        dataset["FIRST_TIME_HOMEBUYER_FLAG"].replace("9", 1, inplace = True)
        dataset["FIRST_TIME_HOMEBUYER_FLAG"].fillna(1, inplace = True)
    else:
        dataset["FIRST_TIME_HOMEBUYER_FLAG"].replace("9", 0, inplace = True)
        dataset["FIRST_TIME_HOMEBUYER_FLAG"].fillna(0, inplace = True)
    
    return dataset

def fillNaN(dataset):
    dataset.fillna(dataset.mean(), inplace =True)
    return dataset

def drop(dataset):
    dataset.drop("SELLER_NAME", axis = 1, inplace = True)
    dataset.drop("SERVICER_NAME", axis = 1, inplace = True)
    return dataset

def convert_PRODUCT_TYPE(dataset):
    dataset["PRODUCT_TYPE"] = dataset["PRODUCT_TYPE"].astype('category')
    dataset["PRODUCT_TYPE"] = dataset["PRODUCT_TYPE"].cat.codes
    return dataset

def convert_LOAN_SEQUENCE_NUMBER(dataset):
    dataset["LOAN_SEQUENCE_NUMBER"] = dataset["LOAN_SEQUENCE_NUMBER"].astype('category')
    dataset["LOAN_SEQUENCE_NUMBER"] = dataset["LOAN_SEQUENCE_NUMBER"].cat.codes
    return dataset

def convert_catogirical_to_numerical(dataset):
    data = pd.get_dummies(dataset, columns=["PROPERTY_STATE","PROPERTY_TYPE","OCCUPANCY_STATUS","CHANNEL","LOAN_PURPOSE"], prefix=["PROPERTY_STATE","PROPERTY_TYPE","OCCUPANCY_STATUS","CHANNEL","LOAN_PURPOSE"])
    return data

def pre_processing(dataset):
    dataset = dropCol(dataset, cutoff = 0.3)
    dataset = drop(dataset)
    dataset = toDatetime(dataset)
    dataset = replace_NaN(dataset)
    dataset = replace_O(dataset)
    dataset = convert_PRODUCT_TYPE(dataset)
    dataset = convert_LOAN_SEQUENCE_NUMBER(dataset)
    dataset = flag_toValue(dataset)
    dataset = convert_catogirical_to_numerical(dataset)
    dataset = fillNaN(dataset)
    #dataset = catogirical_to_numerical(dataset)
    return dataset

def cal_errors(y_testing,y_pred):
    mae = mean_absolute_error(y_testing, y_pred)
    rms = mean_squared_error(y_testing, y_pred)
    r2 = r2_score(y_testing, y_pred)
    mape = mean_absolute_percentage_error(y_testing, y_pred)
    print('MAE = {}, RMS = {}, R2 = {}'.format(mae,rms,r2))

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def scaler(dataset):
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

def nn_modeling(X_training, y_training, X_testing, y_testing):
	for hidden_size in hidden:
    reg = MLPRegressor(solver='adam', alpha=1e-5, hidden_layer_sizes=hidden_size, random_state =1)
    reg.fit(X_training, y_training)
    y_pred = reg.predict(X_testing)
    cal_errors(y_testing,y_pred)

def make_polynomial_regressor(hidden,dgr = 1):
    return make_pipeline(PolynomialFeatures(degree=dgr),
                        MLPRegressor(solver='adam',
                        alpha=1e-5, 
                        hidden_layer_sizes=hidden, 
                        random_state =1))

def nn_polynomial(X_training, y_training, X_testing, y_testing):
	for hidden_size in hidden:
    for d in range(3):
        reg = make_polynomial_regressor(hidden=hidden_size,dgr=d)
        reg.fit(X_training, y_training)
        y_pred = reg.predict(X_testing)
        cal_errors(y_testing,y_pred)



if __name__=='__main__':
	username = input('Please input username:')
	password = input('Please input password:')
	login(username,password)
	if not isLogin():
		raise Exception('Login failed, please check username and password!')
	else:
		print('Congrats! Login successfully!')
	time.sleep(1)
	year_train = input('Please input year:')
	quater_train = input('Please input quater:')
	get_test_date(year_train, quater_train)
	print('year: {}, quater: {}'.format(year_train,quater_train))
	url_train = get_download_link(year_train, quater_train)
	url_test = get_download_link(year_test, quater_test)
	print('start downloading training file for year-{} quater-{}, file name is Q{}{}.zip'.format(year_train,quater_train,quater_train,year_train))
	print('start downloading testing file for year-{} quater-{}, file name is Q{}{}.zip'.format(year_test,quater_test,quater_test,year_test))
	filename_train = 'Q{}{}.zip'.format(quater_train,year_train)
	filename_test = 'Q{}{}.zip'.format(quater_test,year_test)
	download_from_url(url_train,filename_train)
	download_from_url(url_test,filename_test)
	print('downloading finished!')
	un_zip(year_train,quater_train)
	un_zip(year_test,quater_test)
	print('unzipping finished!')

	columns_header_complete = ["CREDIT_SCORE","FIRST_PAYMENT_DATE","FIRST_TIME_HOMEBUYER_FLAG","MATURITY_DATE","MSA","MORTGAGE_INSURANCE_PERCENTAGE","NUMBER_OF_UNITS","OCCUPANCY_STATUS","ORIGINAL_COMBINED_LOAN_TO_VALUE","ORIGINAL_DEBT_TO_INCOME_RATIO","ORIGINAL_UPB","ORIGINAL_LOAN_TO_VALUE","ORIGINAL_INTEREST_RATE","CHANNEL","PREPAYMENT_PENALTY_MORTGAGE_FLAG","PRODUCT_TYPE","PROPERTY_STATE","PROPERTY_TYPE","POSTAL_CODE","LOAN_SEQUENCE_NUMBER","LOAN_PURPOSE","ORIGINAL_LOAN_TERM","NUMBER_OF_BORROWERS","SELLER_NAME","SERVICER_NAME","Pre_HARP_LOAN_SEQUENCE_NUMBER"]
	dataset_train = pd.read_table("'Q{}{}'.format(quater_train,year_train)"+'/'+"Q{}{}.format(quater_train,year_train)"+".txt", header=None, delimiter = "|", names = columns_header_complete)
	dataset_test = pd.read_table("'Q{}{}'.format(quater_test,year_test)"+'/'+"Q{}{}.format(quater_test,year_test)"+".txt", header=None, delimiter = "|", names = columns_header_complete)

	y_training = dataset_train.pop("ORIGINAL_INTEREST_RATE", names = ["ORIGINAL_INTEREST_RATE"])
	y_testing = dataset_test.pop("ORIGINAL_INTEREST_RATE", names = ["ORIGINAL_INTEREST_RATE"])

	X_training = pre_processing(dataset_train)
	X_testing = pre_processing(dataset_test)
	print('Already got the data for training model, good luck!')

	print("Now trainging neural network model.")
	scaler(X_training)
	scaler(y_training)
	scaler(X_testing)
	scaler(y_testing)

	scaler_1 = StandardScaler()
	scaler_1.fit(X_training)
	X_training = scaler_1.transform(X_training)

	scaler_2 = StandardScaler()
	scaler_2.fit(X_testing)
	X_testing = scaler_2.transform(X_testing)
	print("Congrats! Neural network model training has been finished.")
	hidden_layer_size_1 = [87,44]
	hidden_layer_size_2 = [88,70,50,30,10]
	hidden_layer_size_3 = [88,80,70,60,50,40,30,20,10]
	hidden = [hidden_layer_size_1,hidden_layer_size_2,hidden_layer_size_3]

	nn_modeling(X_training, y_training, X_testing, y_testing)
	print("Scores above are the grades for 3 models which have different hypeprameters.")
	nn_polynomial(X_training, y_training, X_testing, y_testing)
	print("Scores above are the grades for models which have different hypeprameters and different polynomial degrees.")

	print("Using forward search to select best features for our model.")
	rf = RandomForestRegressor(max_depth=100)

	sfsSelector = SFS(rf,
                  k_features=15,
                  forward=True,
                  scoring='r2',
                  cv=5)
	sfsSelector = sfsSelector.fit(x_train, y_train)
	features_selected = list(sfs1.k_feature_idx_)
	print("Now we get the best 15 features in this dataset, then we fit our model using these 15 features.")
	
	reg_after = MLPRegressor(solver='adam', alpha=1e-5, hidden_layer_sizes=hidden_size, random_state =1)
    reg_after.fit(X_training[:, features_selected], y_training)
    y_pred_after = reg_after.predict(X_testing[:, features_selected])
    cal_errors(y_testing,y_pred_after)
    print("Now you can compare with the y_pred we got before this part to see how features selection works.")







