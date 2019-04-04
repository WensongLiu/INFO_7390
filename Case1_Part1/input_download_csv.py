
# coding: utf-8

# In[8]:


#pip3 install requests-html
from requests_html import HTMLSession
import re #import regular expression package 正则表达式模块

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
df.to_csv('output_integration_addinput.csv', encoding = 'utf-8', index = False)

