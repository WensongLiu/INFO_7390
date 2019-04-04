import re
import urllib
import urllib.request
import ssl

CIK = '51143'
acc_no = '000005114313000007/0000051143-13-000007'
html_tail = '-index.html'
url_company = "http://www.sec.gov/Archives/edgar/data/"+CIK+"/"+acc_no+html_tail


context = ssl._create_unverified_context()
response = urllib.request.urlopen(url_company,context=context)
res_10q = r'/Archives[\S]+10q.htm"'
lst = re.findall(res_10q,response.read().decode('utf-8'))
url_10q = lst[0]
length = len(url_10q)
url_10q = url_10q[0:length-1]
url_10q = "https://www.sec.gov" + url_10q
response2 = urllib.request.urlopen(url_10q,context=context)
html_page = response2.read().decode('utf-8')

#regex match
reg_table = r'<table.*?</table>'
pattern = re.compile(reg_table,re.S)
tr_lst = pattern.findall(html_page)


print(test_lst)
print("next")
print(tr_lst)
