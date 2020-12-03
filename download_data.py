import urllib.request as request #download from football data
import pandas as pd
import os

def select_league(loc): 
	epl_path = os.getcwd()+'/Data/England/'
	laliga_path = os.getcwd()+'/Data/Spain/'

	if loc == epl_path:
		return 'E0'
	elif loc == laliga_path:
		return 'SP1'

def start(loc):
	print('Downloading...', end='\r')
	file = select_league(loc)
	request.urlretrieve("http://www.football-data.co.uk/mmz4281/2021/"+file+".csv", loc + "20-21.csv")
	request.urlretrieve("http://www.football-data.co.uk/mmz4281/1920/"+file+".csv", loc + "19-20.csv")
	request.urlretrieve("http://www.football-data.co.uk/mmz4281/1819/"+file+".csv", loc + "18-19.csv")
	request.urlretrieve("http://www.football-data.co.uk/mmz4281/1718/"+file+".csv", loc + "17-18.csv")
	request.urlretrieve("http://www.football-data.co.uk/mmz4281/1617/"+file+".csv", loc + "16-17.csv")
	request.urlretrieve("http://www.football-data.co.uk/mmz4281/1516/"+file+".csv", loc + "15-16.csv")
	request.urlretrieve("http://www.football-data.co.uk/mmz4281/1415/"+file+".csv", loc + "14-15.csv")
	request.urlretrieve("http://www.football-data.co.uk/mmz4281/1314/"+file+".csv", loc + "13-14.csv")
	request.urlretrieve("http://www.football-data.co.uk/mmz4281/1213/"+file+".csv", loc + "12-13.csv")
	request.urlretrieve("http://www.football-data.co.uk/mmz4281/1112/"+file+".csv", loc + "11-12.csv")
	request.urlretrieve("http://www.football-data.co.uk/mmz4281/1011/"+file+".csv", loc + "10-11.csv")
	request.urlretrieve("http://www.football-data.co.uk/mmz4281/0910/"+file+".csv", loc + "09-10.csv")
	request.urlretrieve("http://www.football-data.co.uk/mmz4281/0809/"+file+".csv", loc + "08-09.csv")
	request.urlretrieve("http://www.football-data.co.uk/mmz4281/0708/"+file+".csv", loc + "07-08.csv")
	request.urlretrieve("http://www.football-data.co.uk/mmz4281/0607/"+file+".csv", loc + "06-07.csv")
	request.urlretrieve("http://www.football-data.co.uk/mmz4281/0506/"+file+".csv", loc + "05-06.csv")
	request.urlretrieve("http://www.football-data.co.uk/mmz4281/0405/"+file+".csv", loc + "04-05.csv")
	request.urlretrieve("http://www.football-data.co.uk/mmz4281/0304/"+file+".csv", loc + "03-04.csv")
	request.urlretrieve("http://www.football-data.co.uk/mmz4281/0203/"+file+".csv", loc + "02-03.csv")
	request.urlretrieve("http://www.football-data.co.uk/mmz4281/0102/"+file+".csv", loc + "01-02.csv")
	print('DONE', end='\r')

def download_fixtures(url, must=False):
	if not must:
		try:
			request.urlretrieve(url, os.getcwd() + "/Data/fixtures.csv")
		except Exception as e:
			print('\t\tUnable to update fixtures: ', e)
			command = input('Continue or input manually? \n\t\t(c)ontinue\n\t\t(m)anually\n')
			if command == 'm' or command == 'M':
				#bad hack to throw error
				i=j
	else:
		request.urlretrieve(url, os.getcwd() + "/Data/fixtures.csv")

	return pd.read_csv(os.getcwd()+"/Data/fixtures.csv")
