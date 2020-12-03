import prep_training_data
import prep_data_to_predict
import train_predict
import download_data
import warnings
import os

warnings.filterwarnings("ignore")

__banner = """

88                          
88                      @@  
88                        
88,dPPYba,  88   ,aa"'  88  
88P'    "8a  88	"R      88  
88       d8   88        88
88b,   ,a8"   88        88
8Y"Ybbd8"'    88        88  

"""

def select_league(loc): 
	epl_path = os.getcwd()+'/Data/England/'
	laliga_path = os.getcwd()+'/Data/Spain/'

	if loc == 1:
		return epl_path, 'England'
	elif loc == 2:
		return laliga_path, 'Spain'
	else:
		print('Invalid selection: ')

if __name__ == '__main__':
	print(__banner)

	loc = input('Select league: \n\t1. Epl\n\t2. Laliga\n\t')
	arg1, arg2 = select_league(int(loc))
	print('Selected: ', arg2)
	reload_data = input('Download all data online? (Y/N)\n\r')
	if reload_data == 'Y' or reload_data == 'y':
		download_data.start(arg1)
	prep_data_to_predict.start(arg1)
	prep_training_data.start(arg1, arg2)
	train_predict.start(arg1)


