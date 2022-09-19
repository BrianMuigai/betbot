import pandas as pd
import numpy as np
import os
import sys
import progressbar
from prep_training_data import TrainingData
import warnings
import download_data

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

def add_meta_stat(data):
	print('-- calculating stats and metastats', end='\r')
	bar = progressbar.ProgressBar(max_value=len(data))
	table = analyser.create_feature_table(analyser.create_stats_table(data), data)
	table = analyser.create_probs_df(table, analyser.create_poisson_model(table))
	table = analyser.get_last_standings(table, analyser.Standings, '2018')
	bar.update(0)
	HTWS = []
	HTDS = []
	HTLS = []
	ATWS = []
	ATDS = []
	ATLS = []
	HST = []
	AST = []
	HC = []
	AC = []
	HF = []
	AF = []
	HTP = []
	ATP = []
	MR = []
	k = 4
	for i,row in table.iterrows():
		ht = row.HomeTeam
		at = row.AwayTeam

		ht_stats = table[(table.HomeTeam == ht) | (table.AwayTeam == ht)].head(i+1)[-k:]
		at_stats = table[(table.HomeTeam == at) | (table.AwayTeam == at)].head(i+1)[-k:]
		num_games = len(ht_stats)

		home_goals_scored = (ht_stats[ht_stats["AwayTeam"] == ht].sum().FTAG +
			ht_stats[ht_stats["HomeTeam"] == ht].sum().FTHG)/num_games
		home_goals_conceeded = (ht_stats[ht_stats["AwayTeam"] == ht].sum().FTHG +
			ht_stats[ht_stats["HomeTeam"] == ht].sum().FTAG)/num_games
		away_goals_scored = (at_stats[at_stats["AwayTeam"] == at].sum().FTAG +
			at_stats[at_stats["HomeTeam"] == at].sum().FTHG)/num_games
		away_goals_conceeded =  (at_stats[at_stats["AwayTeam"] == at].sum().FTHG +
			at_stats[at_stats["HomeTeam"] == at].sum().FTAG)/num_games
		pastHC = (ht_stats[ht_stats["AwayTeam"] == ht].sum().AC +
			ht_stats[ht_stats["HomeTeam"] == ht].sum().HC)/num_games
		pastAC = (at_stats[at_stats["AwayTeam"] == at].sum().AC +
			at_stats[at_stats["HomeTeam"] == at].sum().HC)/num_games
		pastHST = (ht_stats[ht_stats["AwayTeam"] == ht].sum().AST +
			ht_stats[ht_stats["HomeTeam"] == ht].sum().HST)/num_games
		pastAST = (at_stats[at_stats["AwayTeam"] == at].sum().AST +
			at_stats[at_stats["HomeTeam"] == at].sum().HST)/num_games

		home_rating = (home_goals_scored - home_goals_conceeded)
		away_rating = (away_goals_scored - away_goals_conceeded)
		MR.append(home_rating - away_rating)    
		HC.append(pastHC)
		AC.append(pastAC)
		HST.append(pastHST)
		AST.append(pastAST)
		HTWS.append(analyser.get_team_Winning_streak(analyser.get_team_progress(ht_stats, ht)))
		HTDS.append(analyser.get_team_Drawing_streak(analyser.get_team_progress(ht_stats, ht)))
		HTLS.append(analyser.get_team_loosing_streak(analyser.get_team_progress(ht_stats, ht)))
		ATWS.append(analyser.get_team_Winning_streak(analyser.get_team_progress(at_stats, at)))
		ATDS.append(analyser.get_team_Drawing_streak(analyser.get_team_progress(at_stats, at)))
		ATLS.append(analyser.get_team_loosing_streak(analyser.get_team_progress(at_stats, at)))
		HF.append(analyser.get_form_points(analyser.get_team_progress(ht_stats, ht)))
		AF.append(analyser.get_form_points(analyser.get_team_progress(at_stats, at)))
		HTP.append(analyser.get_form_points(analyser.get_team_progress(table, ht)))
		ATP.append(analyser.get_form_points(analyser.get_team_progress(table, at)))
		bar.update(i)

	table['HTWS'] = HTWS
	table['HTDS'] = HTDS
	table['HTLS'] = HTLS
	table['ATWS'] = ATWS
	table['ATDS'] = ATDS
	table['ATLS'] = ATLS
	table['HST'] = HST
	table['AST'] = AST
	table['HC'] = HC
	table['AC'] = AC
	table['HF'] = HF
	table['AF'] = AF
	table['MR'] = MR
	table['HTP'] = HTP
	table['ATP'] = ATP

	table["CornerDiff"] = (table["HC"] - table["AC"])
	table["ShotsDiff"] = (table["HST"] - table["AST"])
	table["GoalsScoredDiff"] = (table["HGS"] - table["AGS"])
	table["GoalsConceedDiff"] = (table["HGC"] - table["AGC"])
	table["PD"] = (table["HTP"] - table['ATP'])
	table = analyser.get_mw(table)
	table["Result"] = table.apply(lambda row: analyser.transformResult(row),axis=1)
	table["RD"] = (table.HomeTeamLP - table.AwayTeamLP)
	table["DAS"] = (table.HAS - table.AAS)
	table["DDS"] = (table.HDS - table.ADS)

	#rearranging cols
	cols = ['HomeTeam','AwayTeam','FTR','HAS','HDS','HST','AAS','ADS','AST','HC','AC','HTWS','B365H','B365D','B365A',
    'HTDS','HW','HD','HL','AW','AD','AL','WD','DD','LD','HGS','AGS','HGC','AGC','HTLS','ATWS','DAS','DDS',
    'ATDS','ATLS','MR','HF','AF','PD','RD','CornerDiff','ShotsDiff','GoalsScoredDiff','GoalsConceedDiff',
    'HomeTeamLP','AwayTeamLP','HomeTeamCP','AwayTeamCP','probHome','probDraw','probAway','MW','Result']
	table = table[cols]

	return table

def prep_this_weeks_feature(home, away, B365H, B365D, B365A):
	row = [home,away,0,0,'D',0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,B365H,B365D,B365A]
	recent_stats.loc[len(recent_stats) + 1] = row
	

def cols():
	print(recent_stats.columns.values.tolist())

def load_all_teams():
	return list(set(df.tail(380).groupby('HomeTeam').mean().T.columns))

def save_fixture():
	for teams in weeks_teams:
		prep_this_weeks_feature(teams[0], teams[1], teams[2], teams[3], teams[4])
		
	data = add_meta_stat(recent_stats)
	new_fixtures = data.tail(num)
	new_fixtures.to_csv(loc + "this_week.csv", index=False)

def show_all_teams(all_teams):
	i = 1
	for team in all_teams:
		print(i, '\t',team)
		i += 1

def prompt_fixtures():
	global num
	try:
		auto_create_fixture()
	except Exception as e:
		print("Unable to download fixtures: ", e)
		num = int(input('Enter number of matches to predict(max=10): '))
		print('Select teams from this list')
		show_all_teams(load_all_teams())
		for i in range(num):
			get_match(i + 1)

	show_this_week()

def auto_create_fixture():
	global num
	print('Getting fixtures...', end='\r')
	url = 'https://football-data.co.uk/fixtures.csv'
	data = download_data.download_fixtures(url)
	available = data[data.Div == download_data.select_league(loc)]
	if len(available) == 0:
		#bad hack to throw error
		i=j
	num = len(available)
	for index, row in available.iterrows():
		weeks_teams.append([str(row.HomeTeam),str(row.AwayTeam),float(row.B365H),
			float(row.B365D),float(row.B365A)])


def get_match(match):
	all_teams = load_all_teams()
	print('Match: ', str(match))
	homeTeam = input('\tEnter HomeTeam for match: ')
	awayTeam = input('\tEnter AwayTeam for match: ')
	homeWinOdds = input('\tEnter home win odds: ')
	drawOdds = input('\tEnter draw odds: ')
	awayWinOdds = input('\tEnter away win odds: ')
	game = [all_teams[int(homeTeam)-1],all_teams[int(awayTeam) -1],homeWinOdds,drawOdds,awayWinOdds]
	weeks_teams.append(game)

def show_this_week():
	os.system('cls')
	print(__banner)
	print('--------------------------------------------')
	print('\t\tThis weeks fixture: ')
	print('--------------------------------------------')

	i = 1
	for match in weeks_teams:
		print(i, '\t', match[0], '\t\tvs\t\t', match[1])
		i += 1

def start(_loc):
	warnings.filterwarnings("ignore")
	print('============================================')
	print("\t\tENTER THIS WEEKS FIXTURES")
	print('============================================')
	global loc
	global recent_stats
	global weeks_teams
	global new_fixtures
	global analyser
	global df
	loc = _loc

	df = pd.read_csv(loc + '18-19.csv', error_bad_lines=False)
	recent_stats = df.iloc[:, :26]
	recent_stats = recent_stats.drop(['Div','Date'],axis=1)
	try:
		recent_stats = recent_stats.drop(['Referee'], axis=1)
	except:
		recent_stats = recent_stats.iloc[:, :23]
	analyser = TrainingData(loc)
	weeks_teams = []
	new_fixtures = pd.DataFrame()

	prompt_fixtures()
	save_fixture()
