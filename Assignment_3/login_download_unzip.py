import re
import requests
import http.cookiejar
import ssl
import zipfile
import os
import sys

session = requests.session()
session.cookies = http.cookiejar.LWPCookieJar('cookie')

username = input('please input username: ')
password = input('please input password: ')
year = input('please input year: ')
quater = input('please input quater: ')


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
    pattern = re.compile('href=[^<>]*>',re.DOTALL)
    links = pattern.findall(response.text)
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
    url = url[5:len(url)-1]
    download_link = domain + url
    return download_link
    
def download_from_url(url,filename):
	response = session.get(url, headers = headers, allow_redirects = True)
	with open(filename,'wb') as f:
		f.write(response.content)

def un_zip(year,quater):
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
    
def main():
    
    login(username,password)
    if not isLogin():
        raise Exception('Login failed, please check username and password!')
    else:
        print('Congrats! Login successfully!')
    
    print('train dataset -> year: {}, quater: {}'.format(year,quater))
    url = get_download_link(year, quater)
    print('start downloading file for year-{} quater-{}, file name is Q{}{}.zip'.format(year,quater,quater,year))
    filename = 'Q{}{}.zip'.format(quater,year)
    download_from_url(url,filename)
    print('downloading finished! start de-compressing.')
    un_zip(year,quater)
    print('de-compressing finished.')

    
if __name__ == '__main__':
    main()

