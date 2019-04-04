
# coding: utf-8

# In[1]:


import logging

# Using the basicConfig method in logging module
# Create and configure logger
LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
logging.basicConfig(filename = "C:\\Users\\butte\\Jupyter\\Case1_Script\\LoggerTest.log",
                   level = logging.DEBUG,
                   format = LOG_FORMAT)
logger = logging.getLogger()

# Test the logger
# logger.info("Our first message!")


# In[3]:


# pip3 install requests-html
from requests_html import HTMLSession
import re #import regular expression package 正则表达式模块
import zipfile
from zipfile import ZipFile
import os

session = HTMLSession()

# input&read
logger.debug("# Read the CIK and acc_no from console")
cik = input("Please input the CIK：")
acc_no_test = input("Please input the document accession number:")
print('The CIK and Acc_no you entered is:',cik, acc_no_test)

# CIK = '51143'
# acc_no = '000005114313000007/0000051143-13-000007'
logger.debug("# Get the HTML page")
CIK = cik
acc_no = acc_no_test
html_tail = '-index.html'
url_company = "http://www.sec.gov/Archives/edgar/data/"+CIK+"/"+acc_no+html_tail

r1 = session.get(url_company)

url_10q = ""

# match 10q page
logger.debug("# Get the 10Q page")
for url in r1.html.absolute_links:  
	if re.match(r'[a-zA-z]+://[^\s]*.10q.htm',url) != None:
		url_10q = url
		break

# open 10q page
r2 = session.get(url_10q)

# find html element through css selector
# r2.html.find('table') 


#########test.py used to extract all tables in html to csv files

# import requests, BeautifulSoup, pandas
from bs4 import BeautifulSoup
import requests
import pandas as pd

# get all elements from table by BeautifulSoup
# url = 'https://www.sec.gov/Archives/edgar/data/51143/000005114313000007/ibm13q3_10q.htm'
logger.debug("# Read the 10Q HTML page as DataFrame")
url = url_10q
res = requests.get(url)
soup = BeautifulSoup(res.text, 'lxml')
tables = soup.select('table')

# 用pd.read_html直接把HTML中内容读取为DataFrame （8-11行）
# 这一步是关键，pd.read_html方法省去了许多解析HTML的步骤，否则要用BeautifulSoup一个个抓取表格中内容会很繁琐。里面还用到了prettify()方法，可以把BeautifulSoup对象变成字符串，因为pd.read_html处理的是字符串对象

df_list = []
for table in tables:
    df_list.append(pd.concat(pd.read_html(table.prettify())))
df = pd.concat(df_list)

# 最后就是把DataFrame导出到CSV
# df.to_csv('C:\Users\butte\Jupyter\Case1_Script\output.csv',index=False,header=False)
# output the csv file to the destination directory
logger.debug("# Output the DataFrame to csv files")
df.to_csv('C:\\Users\\butte\\Jupyter\\Case1_Script\\csv_files\\output.csv', encoding = 'utf-8', index = False)

print('Output the csv file successfully!')

# compress the csv file to zip file
logger.debug("# Compress the csv file to zip file")
def get_all_files_paths(directory):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)
    return file_paths

def main():
    directory = './csv_files'
    file_paths = get_all_files_paths(directory)
    
    print('Following files will be zipped:')
    for file_name in file_paths:
        print(file_name)
    
    with ZipFile('csv_file.zip', 'w') as zip:
        for file in file_paths:
            zip.write(file)
    print('All files zipped successfully!')

if __name__ == "__main__":
    print ('This is the main of module "Case1_Problem1.py"')
    main()


# In[5]:


# pip install boto3
import boto3
from botocore.client import Config

access_key_id = input('Please input your access_key_id:')
access_secret_key = input('Please input your access_secret_key:')

ACCESS_KEY_ID = access_key_id
ACCESS_SECRET_KEY = access_secret_key

s3 = boto3.client('s3', aws_access_key_id = ACCESS_KEY_ID, 
                   aws_secret_access_key = ACCESS_SECRET_KEY,
                   config = Config(signature_version = 's3v4'))

s3.create_bucket(Bucket = 'group-7')
# s3.upload_file(file_path, bucket_name, file_name)
s3.upload_file('C:\\Users\\butte\\Jupyter\\Case1_Script\\csv_file.zip', 'group-7', 'csv_file.zip')

print('Uploaded successfully!')

