# imports
import urllib.request as rq
import re
import ssl
import zipfile
import os
import random
import numpy as np
import pandas as pds
import logging as lg


#logging functions
def logging_setup():
	# please re-check the filename here, especially the folder path!!!!!!
	lg.basicConfig(filename = './log'+year+'.log',format = '[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',level = lg.DEBUG, filemode = 'a', datefmt = '%Y-%m-%d %I:%M:%S %p')


yr = input("Please input the YEAR：")
# if yr > 2017 or yr < 2003:
# 	lg.critical('Year should between 2003-2017')
# 	os._exit(0)
year = str(yr)
# download .zip file from website
def downloading():

    url_total = 'https://www.sec.gov/files/EDGAR_LogFileData_thru_Jun2017.html'

    ssl._create_default_https_context = ssl._create_unverified_context
    response = rq.urlopen(url_total)

    res_total = r'www.sec.gov/dera/data/Public-EDGAR-log-file-data/'+year+'/Qtr[0-9]/log[0-9]+01\.zip'
    lst = re.findall(res_total,response.read().decode('utf-8'))
    lg.debug('start downloading zip files')
    for i in range(len(lst)):
        url = 'https://'+lst[i]
        month = str(i+1)
        filename = year+"_"+month+"_01.zip"
        rq.urlretrieve(url,filename)
    lg.debug('downloading finished! :D')

# un-zip files
def un_zip():
    lg.debug('start unzip files')
    for i in range(12):
        month = str(i+1)
        file_name = year+'_'+month+'_01.zip'
        folder_path = 'extract_'+year
        zip_file = zipfile.ZipFile(file_name)
        if os.path.isdir(folder_path):
            pass
        else:
            os.mkdir(folder_path)
        for names in zip_file.namelist():
            zip_file.extract(names,folder_path + '/')
        zip_file.close()
    lg.debug('unzip files finished')

row_counts = 0

def read_data_from_filename(filename):
	table = pds.read_csv(filename,low_memory=False)
	return table



def get_most_freq_time(table_copy):
    table_copy['datetime'] = table_copy['date'] + ' ' + table_copy['time']
    pds.to_datetime(table_copy['datetime'])
    df_time_as_index = table_copy.set_index('datetime',drop=True)
    df_time_as_index.index = pds.to_datetime(df_time_as_index.index)
    period = df_time_as_index.to_period('H')
    datetime_freq = pds.value_counts(period.index)
    most_freq_datetime = datetime_freq.to_timestamp().index[0]
    return most_freq_datetime

def get_random_zero_or_one(column_name,table_copy):
    total_counts = pds.value_counts(table_copy[column_name])
    ratio_of_index1 = total_counts[0]/ len(table_copy.index)
    random_number = random.randint(0,10)
    if random_number <= ratio_of_index1 * 10:
        return total_counts.index[0]
    else:
        return 1 - total_counts.index[0]
    
def replaceAll_outlier_withNaN(table_copy):
    table_copy['noagent'] = np.where(np.abs(table_copy['noagent'] - 0.5) != 0.5,np.nan,table_copy['noagent'])
    table_copy['norefer'] = np.where(np.abs(table_copy['norefer'] - 0.5) != 0.5,np.nan,table_copy['norefer'])
    table_copy['crawler'] = np.where(np.abs(table_copy['crawler'] - 0.5) != 0.5,np.nan,table_copy['crawler'])
    table_copy['idx'] = np.where(np.abs(table_copy['idx'] - 0.5) != 0.5,np.nan,table_copy['idx'])
    table_copy['size'] = np.where(table_copy['size'] > 10000000,np.nan,table_copy['size'])
    #find outliers
    list_0_to_10 = np.arange(11)
    check_find = np.logical_not(np.isin(table_copy['find'],list_0_to_10))
    table_copy['find'] = np.where(check_find,np.nan,table_copy['find'])
    #browser outliers
    list_browser = ['mie','fox','saf','chr','sea','opr','oth','win','mac','lin','iph','ipd','and','rim','iem']
    check_browser = np.logical_not(np.isin(table_copy['browser'],list_browser))
    table_copy['browser'] = np.where(check_browser,np.nan,table_copy['browser'])
    #code outliers
    list_httpcode = [100,101,102,200,201,202,203,204,205,206,207,208,226,300,301,302,303,304,305,306,307,308,400,401,402,403,404,405,406,407,408,409,410,411,412,413,414,415,416,417,418,420,421,422,423,424,425,426,428,429,431,444,450,451,494,500,501,502,503,504,505,506,507,508,510,511]
    check_code = np.logical_not(np.isin(table_copy['code'],list_httpcode))
    table_copy['code'] = np.where(check_code,np.nan,table_copy['code'])
    return table_copy
    
def drop_replace_nan(table_copy):
    table_copy = table_copy.dropna(subset=['ip','cik','accession','extention'])
    mean_of_size = np.mean(table_copy['size'])
    fill_values = {'datetime':get_most_freq_time(table_copy),'zone':0,'code':200,'size': mean_of_size,'idx':1,'browser':'ukn','norefer':get_random_zero_or_one('norefer',table_copy),'noagent':get_random_zero_or_one('noagent',table_copy),'crawler':get_random_zero_or_one('crawler',table_copy),'find':0}
    table_copy = table_copy.fillna(value=fill_values)
    return table_copy

def main():
    logging_setup()
    #download zip files
    # downloading()

    # #un_zip files into folder: ./extract_year/...
    # un_zip()

    print('data download and un_zip finished!!')
    print('will start data cleaning now')

    #create a list of filenames according to year, not finished
    #be careful about the path of filename!!!!!!!!!!
    #specify the file path here with its filename, e.g. : ./folder_name/folder_name/filename
    filename_list = list()

    for i in range(12):
        month = '{:0>2d}'.format(i+1)
        filename = 'log'+year+month+'01.csv'
        filename_list.append(filename)

    #loop through the filename_list and do the data cleaning
    for filename in filename_list:

        print('start data cleaning for file: ' + filename)
        lg.debug('  start data cleaning for file: ' + filename)
        #read data first
        table_copy = read_data_from_filename('./extract_2016/'+filename)

        
        #replace all outliers with NaN
        print('start replacing outliers')
        lg.debug('start replacing outliers')
        table_copy = replaceAll_outlier_withNaN(table_copy)

        #remove or replace NaN values
        print('start replacing NaN')
        lg.debug('start replacing NaN')
        table_copy = drop_replace_nan(table_copy)

        print('finished data cleaning for fils: '+filename)
        lg.debug('  finished data cleaning for file: ' + filename)
        # delete date column and time column
        # not finished

        # save the table into a new csv file
        # not finished


if __name__ == '__main__':
    main()































