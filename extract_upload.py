#pip3 install requests-html
from requests_html import HTMLSession
import re

session = HTMLSession()


CIK = '51143'
acc_no = '000005114313000007/0000051143-13-000007'
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
r2.html.find('table') 
