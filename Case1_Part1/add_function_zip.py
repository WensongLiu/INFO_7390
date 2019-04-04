
# coding: utf-8

# In[1]:


#pip3 install requests-html
from requests_html import HTMLSession
import re #import regular expression package 正则表达式模块
import zipfile
from zipfile import ZipFile
import os

session = HTMLSession()


##input&read

cik = input("Please input the CIK：")
acc_no_test = input("Please input the document accession number:")
print(cik, acc_no_test)

#CIK = '51143'
#acc_no = '000005114313000007/0000051143-13-000007'
CIK = cik
acc_no = acc_no_test
html_tail = '-index.html'
url_company = "http://www.sec.gov/Archives/edgar/data/"+CIK+"/"+acc_no+html_tail

r1 = session.get(url_company)

url_10q = ""
# match 10q page
for url in r1.html.absolute_links:  
	if re.match(r'[a-zA-z]+://[^\s]*.10q.htm',url) != None:
		url_10q = url
		break

#open 10q page
r2 = session.get(url_10q)

# find html element through css selector
#r2.html.find('table') 


#########test.py

# import requests, BeautifulSoup, pandas
from bs4 import BeautifulSoup
import requests
import pandas as pd

#get all elements from table by BeautifulSoup

#url = 'https://www.sec.gov/Archives/edgar/data/51143/000005114313000007/ibm13q3_10q.htm'

url = url_10q
res = requests.get(url)
soup = BeautifulSoup(res.text, 'lxml')
tables = soup.select('table')

#用pd.read_html直接把HTML中内容读取为DataFrame （8-11行）
#这一步是关键，pd.read_html方法省去了许多解析HTML的步骤，否则要用BeautifulSoup一个个抓取表格中内容会很繁琐。里面还用到了prettify()方法，可以把BeautifulSoup对象变成字符串，因为pd.read_html处理的是字符串对象

df_list = []
for table in tables:
    df_list.append(pd.concat(pd.read_html(table.prettify())))
df = pd.concat(df_list)

#最后就是把DataFrame导出到CSV
# df.to_csv('C:\Users\butte\Jupyter\Case1_Script\output.csv',index=False,header=False)

#output the csv file to the destination directory
df.to_csv('C:\\Users\\butte\\Jupyter\\Case1_Script\\csv_files\\output.csv', encoding = 'utf-8', index = False)

print('Output the csv file successfully!')

#compress the csv file to zip file
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
    print ('This is the main of module "integration.py"')
    main()

