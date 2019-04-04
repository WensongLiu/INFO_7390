# import requests, BeautifulSoup, pandas
# BeautifulSoup provides some simple ways to get data from website.
# Requests is a http library, it makes web developing easier and safer.
# Pandas is an open source Python data analysis library(completeness of its data analysis functions, high speed for big data operations).

from bs4 import BeautifulSoup
import requests
import pandas as pd

#select all tables from a specific website and get all elements from these tables by BeautifulSoup
#there're 4 different beautifulsoup parser(html.parser, lxml, lxml-xml, html5lib)
url = 'https://ucr.fbi.gov/crime-in-the-u.s/2016/crime-in-the-u.s.-2016/topic-pages/tables/table-1'
res = requests.get(url)
soup = BeautifulSoup(res.text, 'lxml')
tables = soup.select('table')

#read HTML content as DataFrame by function called read_html provided by pandas
#pd.read_html is a simple way to parse html
#function prettify() can transfrom BeautifulSoup's data type to String，because the arguments of function pd.read_html must be String
df_list = []
for table in tables:
    df_list.append(pd.concat(pd.read_html(table.prettify())))
df = pd.concat(df_list)

#export csv.file
#the whole to_csv function is writen like df.to_csv('path',index = ... , header = ...)
df.to_csv('/Users/G/Documents/2018_Fall/DataScience/GitProject_7390/dataExtraction.csv',index = False)
    
#We can use this method to export huge size dataset with a nite and clean form from a specific html website automatically, rapidly and safely.
