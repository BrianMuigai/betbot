from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier#for away|draw
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

clf3 = XGBClassifier(n_estimators=50)
clf4 = SVC(kernel='rbf', random_state=912)
clf6 = LogisticRegression(C=8,solver='lbfgs',multi_class='ovr')

loc = ''
thisweek_matches = None
this_week = None
dataset = None
X_all = None
y_all = None
X_test = None

scaler = StandardScaler()
X_feats = ['HST','AST','HAS','HDS','AAS','ADS','HC','AC','HTWS','HTDS','HTLS','ATWS','ATDS','ATLS',
'HW','HD','HL','AW','AD','AL','HGS','AGS','HGC','AGC','WD','DD','LD','HF','AF','MR','MW','CornerDiff',
'GoalsScoredDiff','GoalsConceedDiff','ShotsDiff','HomeTeamLP','AwayTeamLP','PD','RD','DAS','DDS'
,'B365H','B365D','B365A','probHome','probDraw','probAway','HomeTeamCP','AwayTeamCP',]

def load_stats():
	global this_week
	global thisweek_matches
	global dataset
	global X_all
	global y_all
	global X_test

	dataset = pd.read_csv(loc + 'training_dataset.csv')
	thisweek_matches = pd.read_csv(loc + 'this_week.csv')
	this_week = pd.DataFrame()
	this_week['HomeTeam'] = thisweek_matches.HomeTeam
	this_week['AwayTeam'] = thisweek_matches.AwayTeam
	this_week['B365H'] = thisweek_matches.B365H
	this_week['B365D'] = thisweek_matches.B365D
	this_week['B365A'] = thisweek_matches.B365A

	X_all = dataset[X_feats]
	scaler.fit(X_all)
	X_all = scaler.transform(X_all)
	y_all = dataset['Result']

	X_test = thisweek_matches[X_feats]
	X_test = scaler.transform(X_test)

def transform_result(result):
    '''Converts results (H,A or D) into numeric values'''
    if result == 1:
        return 'H'
    elif result == 0:
        return 'D'
    else:
        return 'A'

def train():
	clf3.fit(X_all, y_all)
	clf4.fit(X_all, y_all)
	clf6.fit(X_all, y_all)

	scores = []
	scores.append(clf3.score(X_all, y_all))
	scores.append(clf4.score(X_all, y_all))
	scores.append(clf6.score(X_all, y_all))

	print('\t\tSCORES\n\t clf3: ', scores[0],' clf4: ', scores[1],' clf6: ', scores[2])

def predict():
	clf3_pred = clf3.predict(X_test)
	clf4_pred = clf4.predict(X_test)
	clf6_pred = clf6.predict(X_test)
	prediction = []
	for i in range(len(X_test)):
		tmp = set()
		if clf3_pred[i] != clf6_pred[i]:
			tmp.add(transform_result(clf3_pred[i]))
		if clf4_pred[i] != clf6_pred[i]:
			tmp.add(transform_result(clf4_pred[i]))
		
		tmp.add('*' + transform_result(clf6_pred[i]) + '*')
		
		prediction.append(','.join(list(tmp)))

	return prediction

def save_predictions(prediction):
	this_week['Prediction'] = prediction
	this_week.to_csv(loc + 'predictions.csv', index=False)
	print(this_week)

def start(_loc):
	warnings.filterwarnings("ignore")
	print('======================================================')
	print('\t\tTRAINING AND PREDICTION')
	print('======================================================')
	global loc
	loc = _loc
	load_stats()
	train()
	print('saving prediction...', end='\r')
	prediction = predict()
	save_predictions(prediction)
	print('DONE', end='\r')



	
