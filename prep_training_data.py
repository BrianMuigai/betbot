import pandas as pd
import numpy as np
import scipy.stats as scipy
import matplotlib.pyplot as plt
import urllib.request as request #download from football data
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson,skellam
import progressbar

import warnings
warnings.filterwarnings("ignore")

class TrainingData(object):
	"""docstring for TrainingData"""
	warnings.filterwarnings("ignore")

	def __init__(self, _loc):
		super(TrainingData, self).__init__()
		self.loc = _loc
		self.load_data()

	def set_loc(self, _loc):
		self.loc = _loc

	def get_mw(self, playing_stat):
		MatchWeek = []
		for i in range(len(playing_stat)):
			week = int(len(playing_stat.head(i)) / 10)
			MatchWeek.append(week)
		playing_stat['MW'] = MatchWeek

		return playing_stat

	def create_stats_table(self, df):
		#Team, Home Goals Score, Away Goals Score, Attack Strength, Home Goals Conceded, Away Goals Conceded, Defensive Strength
		stats_table = pd.DataFrame(columns=('Team','HGS','AGS','HAS','AAS','HGC','AGC','HDS','ADS'))

		avg_home_scored = df.FTHG.sum()*1.0 / df.shape[0]
		avg_away_scored = df.FTAG.sum()*1.0 / df.shape[0]
		avg_home_conceded = avg_away_scored
		avg_away_conceded = avg_home_scored

		res_home = df.groupby('HomeTeam')
		res_away = df.groupby('AwayTeam')

		stats_table.Team = res_home.HomeTeam.apply(pd.DataFrame).columns.values
		try:
			stats_table.HGS = res_home.FTHG.sum().values
		except Exception as e:
			stats_table.HGS = [0 for i in range(20)]
		try:
			stats_table.HGC = res_home.FTAG.sum().values
		except Exception as e:
			stats_table.HGC = [0 for i in range(20)]
		try:
			stats_table.AGS = res_away.FTAG.sum().values
		except Exception as e:
			stats_table.AGS = [0 for i in range(20)]
		try:
			stats_table.AGC = res_away.FTHG.sum().values
		except Exception as e:
			stats_table.AGC = [0 for i in range(20)]

		#Assuming number of home games = number of away games
		num_games = df.shape[0]/20
		stats_table.HAS = (stats_table.HGS / num_games) / avg_home_scored
		stats_table.AAS = (stats_table.AGS / num_games) / avg_away_scored
		stats_table.HDS = (stats_table.HGC / num_games) / avg_home_conceded
		stats_table.ADS = (stats_table.AGC / num_games) / avg_away_conceded

		return stats_table

	def count_prog(self, team_prog):
		wins = 0
		draws = 0
		loses = 0
		num_games = len(team_prog)

		for outcome in team_prog:
			if outcome == 'W':
				wins += 1
			elif outcome == 'D':
				draws += 1
			else:
				loses += 1
		return wins/num_games, draws/num_games, loses/num_games
	''' feature_table contains all the fixtures in the current season.
	ftr = full time result
	hst = home shots on target
	ast = away shots on target
	'''
	def create_feature_table(self, stats, data):
		feature_table = data[['HomeTeam','AwayTeam','FTR','FTHG','FTAG','HST','AST', 'HC', 'AC','B365H','B365D','B365A']]
		f_HAS = []
		f_HDS = []
		f_AAS = []
		f_ADS = []
		f_HGS = []
		f_HGC = []
		f_AGS = []
		f_AGC = []
		HW = []
		HD = []
		HL = []
		AW = []
		AD = []
		AL = []
		for index,row in feature_table.iterrows():
			f_HAS.append(stats[stats['Team'] == row['HomeTeam']]['HAS'].values[0])
			f_HDS.append(stats[stats['Team'] == row['HomeTeam']]['HDS'].values[0])
			f_AAS.append(stats[stats['Team'] == row['AwayTeam']]['AAS'].values[0])
			f_ADS.append(stats[stats['Team'] == row['AwayTeam']]['ADS'].values[0])
			f_HGS.append(stats[stats['Team'] == row['AwayTeam']]['HGS'].values[0])
			f_HGC.append(stats[stats['Team'] == row['AwayTeam']]['HGC'].values[0])
			f_AGS.append(stats[stats['Team'] == row['AwayTeam']]['AGS'].values[0])
			f_AGC.append(stats[stats['Team'] == row['AwayTeam']]['AGC'].values[0])

			h_progress_stats = self.count_prog(self.get_team_progress(feature_table, row.HomeTeam))
			a_progress_stats = self.count_prog(self.get_team_progress(feature_table, row.AwayTeam))
			HW.append(h_progress_stats[0])
			HD.append(h_progress_stats[1])
			HL.append(h_progress_stats[2])
			AW.append(a_progress_stats[0])
			AD.append(a_progress_stats[1])
			AL.append(a_progress_stats[2])

		feature_table['HAS'] = f_HAS
		feature_table['HDS'] = f_HDS
		feature_table['AAS'] = f_AAS
		feature_table['ADS'] = f_ADS
		feature_table['HGS'] = f_HGS
		feature_table['HGC'] = f_HGC
		feature_table['AGS'] = f_AGS
		feature_table['AGC'] = f_AGC

		feature_table['HW'] = HW
		feature_table['HD'] = HD
		feature_table['HL'] = HL
		feature_table['AW'] = AW
		feature_table['AD'] = AD
		feature_table['AL'] = AL

		feature_table["WD"] = feature_table.HW - feature_table.AW
		feature_table["DD"] = feature_table.HD - feature_table.AD
		feature_table["LD"] = feature_table.HL - feature_table.AL
		
		feature_table = self.get_mw(feature_table)

		return feature_table

	def load_standings(self):
		self.Standings = pd.read_csv(self.loc + "Standings.csv")
		self.Standings.set_index(['Team'], inplace=True)
		self.Standings = self.Standings.fillna(18)

	def get_team_points(self, team, data):
		points = 0
		for i in range(len(data)):
			if data.iloc[i].HomeTeam == team or data.iloc[i].AwayTeam == team:
				if data.iloc[i].HomeTeam == team and data.iloc[i].FTHG > data.iloc[i].FTAG:
					points += 3
				elif data.iloc[i].AwayTeam == team and data.iloc[i].FTAG > data.iloc[i].FTHG:
					points +=3
				elif data.iloc[i].FTHG == data.iloc[i].FTAG:
					points += 1

		return points

	def get_leaders_board(self, data):
		print("-- getting leaders board")
		league_participants = data.groupby('HomeTeam').mean().T.columns
		leaders_board = []
		for team in league_participants:
			row = [str(team), self.get_team_points(team, data)]
			leaders_board.append(row)    

		leaders_board = sorted(leaders_board, key = lambda leaders_board : leaders_board[1], reverse = True)
		print("--done getting leaders board")
		return leaders_board

	def get_pos(self, team, leaders_board):
		pos = None
		count = 0
		for row in leaders_board:
			count +=1
			if row[0] == team:
				pos = count
				return pos
		return pos

	def get_new_teams(self, ateams, bteams):
		in_ateams = set(ateams)
		in_bteams = set(bteams)
		in_bteams_but_not_in_ateams = in_bteams - in_ateams
		new_teams = list(in_bteams_but_not_in_ateams)
		return new_teams

	def initialize_teams(self, data, standings):
		current_teams = data.groupby('HomeTeam').mean().T.columns
		all_teams = standings.Team
		new_teams = self.get_new_teams(all_teams, current_teams)
		if len(new_teams) > 0:
			for team in new_teams:
				row = []
				row.append(team)
			for cols in range(standings.shape[1] - 1):
				row.append(None)
				standings.loc[len(standings) + 1] = row
				print("new row", row)
			standings.to_csv(self.loc + "Standings.csv", index = False)

	def save_standings(self, data, yr):
		print('Saving standings...')

		standings = None
		try:
			standings = pd.read_csv(self.loc + "Standings.csv")
		except:
			self.create_standings_table()
			self.save_standings(data, yr)
		standings = pd.read_csv(self.loc + "Standings.csv")
		current_teams = data.groupby('HomeTeam').mean().T.columns
		all_teams = standings.Team
		leaders_board = self.get_leaders_board(data)
		pos = []
		for team in all_teams:
			pos.append(self.get_pos(team, leaders_board))
		standings[yr] = pos  
		standings.to_csv(self.loc + "Standings.csv", index=False)

	def create_standings_table(self):
		datas = [self.raw_data_5,self.raw_data_6,self.raw_data_7,self.raw_data_8,self.raw_data_9,
		 self.raw_data_10,self.raw_data_11,self.raw_data_12,self.raw_data_13,self.raw_data_14,self.raw_data_15,
		 self.raw_data_16,self.raw_data_17,self.raw_data_18, self.raw_data_19]
		df = pd.concat(datas, ignore_index=True)
		data = df.groupby('HomeTeam')
		standings = pd.DataFrame()
		standings['Team'] = data.HomeTeam.apply(pd.DataFrame).columns.values
		standings.to_csv(self.loc + 'Standings.csv', index=False)
		yr = 2005
		for data in datas:
			self.save_standings(data, str(yr))
			yr += 1

	def get_last_standings(self, playing_stat, Standings, year):
		HomeTeamLP = []
		HomeTeamCP = []
		AwayTeamLP = []
		AwayTeamCP = []
		for i in range(len(playing_stat)):
			ht = playing_stat.iloc[i].HomeTeam
			at = playing_stat.iloc[i].AwayTeam
			HomeTeamLP.append(Standings.loc[ht][year])
			HomeTeamCP.append(Standings.loc[ht][str(int(year)+1)])
			AwayTeamLP.append(Standings.loc[at][year])
			AwayTeamCP.append(Standings.loc[at][str(int(year)+1)])
		playing_stat['HomeTeamLP'] = HomeTeamLP
		playing_stat['HomeTeamCP'] = HomeTeamCP
		playing_stat['AwayTeamLP'] = AwayTeamLP
		playing_stat['AwayTeamCP'] = AwayTeamCP
		return playing_stat

	def get_team_progress(self, df, team):
		df = df[(df.HomeTeam == team) | (df.AwayTeam == team)]
		data = np.array(df)
		progress = []
		for row in data:
			if row[0] == team:
				if row[2] == 'H':
					progress.append("W")
				elif row[2] == 'D':
					progress.append("D")
				elif row[2] == 'A':
					progress.append("L")

			elif row[1] == team:
				if row[2] == 'A':
					progress.append("W")
				elif row[2] == 'D':
					progress.append("D")
				elif row[2] == 'H':
					progress.append("L")

		return progress
	    
	def get_team_loosing_streak(self, team_prog):
		team_loosing_streak = 0
		tmp = 0

		for i in team_prog:
			if i == "L":
				tmp += 1
			elif tmp > team_loosing_streak:
				team_loosing_streak = tmp
				tmp = 0
			else:
				tmp = 0

		return team_loosing_streak

	def get_team_Drawing_streak(self, team_prog):
		team_loosing_streak = 0
		tmp = 0

		for i in team_prog:
			if i == "D":
				tmp += 1
			elif tmp > team_loosing_streak:
				team_loosing_streak = tmp
				tmp = 0
			else:
				tmp = 0

		return team_loosing_streak

	def get_team_Winning_streak(self, team_prog):
		team_loosing_streak = 0
		tmp = 0

		for i in team_prog:
			if i == "W":
				tmp += 1
			elif tmp > team_loosing_streak:
				team_loosing_streak = tmp
				tmp = 0
			else:
				tmp = 0

		return team_loosing_streak

	def get_form_points(self, team_prog):
		point = 0
		for result in team_prog:
			if result == 'W':
				point += 3
			elif result == 'D':
				point +=1

		return point

	def create_poisson_model(self, df):
		goal_model_data = pd.concat([df[['HomeTeam','AwayTeam','FTHG']].assign(home=1).rename(
			columns={'HomeTeam':'HomeTeam', 'AwayTeam':'AwayTeam','FTHG':'goals'}),
		df[['AwayTeam','HomeTeam','FTAG']].assign(home=0).rename(
			columns={'AwayTeam':'HomeTeam', 'HomeTeam':'AwayTeam','FTAG':'goals'})])
		goal_model_data[['goals']] = goal_model_data[['goals']].astype(int)
		poisson_model = smf.glm(formula='goals ~ home + HomeTeam + AwayTeam', data=goal_model_data,
			family=sm.families.Poisson()).fit()
		return poisson_model

	def simulate_match(self,foot_model, homeTeam, awayTeam, max_goals=5):
		home_goals_avg = foot_model.predict(pd.DataFrame(data={'HomeTeam': homeTeam, 
			'AwayTeam': awayTeam,'home':1},index=[1])).values[0]
		away_goals_avg = foot_model.predict(pd.DataFrame(data={'HomeTeam': awayTeam, 
			'AwayTeam': homeTeam,'home':0},index=[1])).values[0]
		team_pred = [[poisson.pmf(
			i, team_avg) for i in range(0, max_goals+1)] for team_avg in [home_goals_avg, away_goals_avg]]
		return(np.outer(np.array(team_pred[0]), np.array(team_pred[1])))

	def create_probs_df(self, df, model):
		probHome = []
		probDraw = []
		probAway = []
		for i,row in df.iterrows():
			stats = self.simulate_match(model, row.HomeTeam, row.AwayTeam)
			probHome.append(np.sum(np.tril(stats, -1)))
			probDraw.append(np.sum(np.diag(stats)))
			probAway.append(np.sum(np.triu(stats, 1)))
		df['probHome'] = probHome
		df['probDraw'] = probDraw
		df['probAway'] = probAway

		return df

	def add_meta_stat(self, table, yr):
		print('-- calculating stats and metastats', end='\r')
		bar = progressbar.ProgressBar(max_value=len(table))
		table = self.create_feature_table(self.create_stats_table(table), table)
		table = self.create_probs_df(table, self.create_poisson_model(table))
		table = self.get_last_standings(table, self.Standings, yr)
		bar.update(0)
		HTWS = []
		HTDS = []
		HTLS = []
		ATWS = []
		ATDS = []
		ATLS = []
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

			home_rating = (home_goals_scored - home_goals_conceeded)
			away_rating = (away_goals_scored - away_goals_conceeded)
			MR.append(home_rating - away_rating)
			HTWS.append(self.get_team_Winning_streak(self.get_team_progress(ht_stats, ht)))
			HTDS.append(self.get_team_Drawing_streak(self.get_team_progress(ht_stats, ht)))
			HTLS.append(self.get_team_loosing_streak(self.get_team_progress(ht_stats, ht)))
			ATWS.append(self.get_team_Winning_streak(self.get_team_progress(at_stats, at)))
			ATDS.append(self.get_team_Drawing_streak(self.get_team_progress(at_stats, at)))
			ATLS.append(self.get_team_loosing_streak(self.get_team_progress(at_stats, at)))
			HF.append(self.get_form_points(self.get_team_progress(ht_stats, ht)))
			AF.append(self.get_form_points(self.get_team_progress(at_stats, at)))
			HTP.append(self.get_form_points(self.get_team_progress(table, ht)))
			ATP.append(self.get_form_points(self.get_team_progress(table, at)))
			bar.update(i+1)

		table['HTWS'] = HTWS
		table['HTDS'] = HTDS
		table['HTLS'] = HTLS
		table['ATWS'] = ATWS
		table['ATDS'] = ATDS
		table['ATLS'] = ATLS
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
		table = self.get_mw(table)
		table["Result"] = table.apply(lambda row: self.transformResult(row),axis=1)
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

	def transformResult(self, row):
		'''Converts results (H,A or D) into numeric values'''
		if(row.FTR == 'H'):
			return 1
		elif(row.FTR == 'A'):
			return -1
		else:
			return 0

	def load_data(self):
		print('Reading data...', end='\r')
		self.raw_data_5 = pd.read_csv(self.loc + '04-05.csv', error_bad_lines=False, encoding= 'unicode_escape')[:380]
		self.raw_data_6 = pd.read_csv(self.loc + '05-06.csv', error_bad_lines=False, encoding= 'unicode_escape')[:380]
		self.raw_data_7 = pd.read_csv(self.loc + '06-07.csv', error_bad_lines=False, encoding= 'unicode_escape')[:380]
		self.raw_data_8 = pd.read_csv(self.loc + '07-08.csv', error_bad_lines=False, encoding= 'unicode_escape')[:380]
		self.raw_data_9 = pd.read_csv(self.loc + '08-09.csv', error_bad_lines=False, encoding= 'unicode_escape')[:380]
		self.raw_data_10 = pd.read_csv(self.loc + '09-10.csv', error_bad_lines=False, encoding= 'unicode_escape')[:380]
		self.raw_data_11 = pd.read_csv(self.loc + '10-11.csv', error_bad_lines=False, encoding= 'unicode_escape')[:380]
		self.raw_data_12 = pd.read_csv(self.loc + '11-12.csv', error_bad_lines=False, encoding= 'unicode_escape')[:380]
		self.raw_data_13 = pd.read_csv(self.loc + '12-13.csv', error_bad_lines=False, encoding= 'unicode_escape')[:380]
		self.raw_data_14 = pd.read_csv(self.loc + '13-14.csv', error_bad_lines=False, encoding= 'unicode_escape')[:380]
		self.raw_data_15 = pd.read_csv(self.loc + '14-15.csv', error_bad_lines=False, encoding= 'unicode_escape')[:380]
		self.raw_data_16 = pd.read_csv(self.loc + '15-16.csv', error_bad_lines=False, encoding= 'unicode_escape')[:380]
		self.raw_data_17 = pd.read_csv(self.loc + '16-17.csv', error_bad_lines=False, encoding= 'unicode_escape')[:380]
		self.raw_data_18 = pd.read_csv(self.loc + '17-18.csv', error_bad_lines=False, encoding= 'unicode_escape')[:380]
		self.raw_data_19 = pd.read_csv(self.loc + '18-19.csv', error_bad_lines=False, encoding= 'unicode_escape')[:380]
		self.save_standings(self.raw_data_19, '2019')
		self.load_standings()

	def add_stats(self):
		self.feature_table_6 = self.add_meta_stat(self.raw_data_6, '2005')
		self.feature_table_6.to_csv(self.loc + 'stats/06.csv', index=False)

		self.feature_table_7 = self.add_meta_stat(self.raw_data_7, '2006')
		self.feature_table_7.to_csv(self.loc + 'stats/07.csv', index=False)

		self.feature_table_8 = self.add_meta_stat(self.raw_data_8, '2007')
		self.feature_table_8.to_csv(self.loc + 'stats/08.csv', index=False)

		self.feature_table_9 = self.add_meta_stat(self.raw_data_9, '2008')
		self.feature_table_9.to_csv(self.loc + 'stats/09.csv', index=False)

		self.feature_table_10 = self.add_meta_stat(self.raw_data_10, '2009')
		self.feature_table_10.to_csv(self.loc + 'stats/10.csv', index=False)

		self.feature_table_11 = self.add_meta_stat(self.raw_data_11, '2010')
		self.feature_table_11.to_csv(self.loc + 'stats/11.csv', index=False)

		self.feature_table_12 = self.add_meta_stat(self.raw_data_12, '2011')
		self.feature_table_12.to_csv(self.loc + 'stats/12.csv', index=False)

		self.feature_table_13 = self.add_meta_stat(self.raw_data_13, '2012')
		self.feature_table_13.to_csv(self.loc + 'stats/13.csv', index=False)

		self.feature_table_14 = self.add_meta_stat(self.raw_data_14, '2013')
		self.feature_table_14.to_csv(self.loc + 'stats/14.csv', index=False)

		self.feature_table_15 = self.add_meta_stat(self.raw_data_15, '2014')
		self.feature_table_15.to_csv(self.loc + 'stats/15.csv', index=False)

		self.feature_table_16 = self.add_meta_stat(self.raw_data_16, '2015')
		self.feature_table_16.to_csv(self.loc + 'stats/16.csv', index=False)

		self.feature_table_17 = self.add_meta_stat(self.raw_data_17, '2016')
		self.feature_table_17.to_csv(self.loc + 'stats/17.csv', index=False)

		self.feature_table_18 = self.add_meta_stat(self.raw_data_18, '2017')
		self.feature_table_18.to_csv(self.loc + 'stats/18.csv', index=False)

		self.feature_table_19 = self.add_meta_stat(self.raw_data_19, '2018')
		self.feature_table_19.to_csv(self.loc + 'stats/19.csv', index=False)

	def add_stats_latest(self):
		self.feature_table_6 = pd.read_csv(self.loc + 'stats/06.csv')
		self.feature_table_7 = pd.read_csv(self.loc + 'stats/07.csv')
		self.feature_table_8 = pd.read_csv(self.loc + 'stats/08.csv')
		self.feature_table_9 = pd.read_csv(self.loc + 'stats/09.csv')
		self.feature_table_10 = pd.read_csv(self.loc + 'stats/10.csv')
		self.feature_table_11 = pd.read_csv(self.loc + 'stats/11.csv')
		self.feature_table_12 = pd.read_csv(self.loc + 'stats/12.csv')
		self.feature_table_13 = pd.read_csv(self.loc + 'stats/13.csv')
		self.feature_table_14 = pd.read_csv(self.loc + 'stats/14.csv')
		self.feature_table_15 = pd.read_csv(self.loc + 'stats/15.csv')
		self.feature_table_16 = pd.read_csv(self.loc + 'stats/16.csv')
		self.feature_table_17 = pd.read_csv(self.loc + 'stats/17.csv')
		self.feature_table_18 = pd.read_csv(self.loc + 'stats/18.csv')

		self.feature_table_19 = self.add_meta_stat(self.raw_data_19, '2018')
		self.feature_table_19.to_csv(self.loc + 'stats/19.csv', index=False)


	def create_df(self):
		final_df  = pd.concat([self.feature_table_6,
					self.feature_table_7,
					self.feature_table_8,
					self.feature_table_9,
					self.feature_table_10,
					self.feature_table_11,
					self.feature_table_12,
					self.feature_table_13,
					self.feature_table_14,
					self.feature_table_15,
					self.feature_table_16,
					self.feature_table_17,
					self.feature_table_18,
					self.feature_table_19], ignore_index=True)
		
		final_df.to_csv(self.loc + 'training_dataset.csv', index=False)


def start(_loc, league):
	print('============================================')
	print('\t\tSCRAPPING AND CLEANING DATA')
	print('============================================')

	print('Preparing model dataset')
	loc = "C:/Users/Naomi/Desktop/personal/Locker/ML_project/research/epl-prediction-2017/console/Data/England/"
	download_url = 'http://www.football-data.co.uk/mmz4281/1819/'

	loc = _loc
	if league == 'England':
		download_url += 'E0'
	else:
		download_url += 'SP1'

	trainingData = TrainingData(loc)

	def download_data():
		request.urlretrieve(download_url, loc+"18-19.csv")

	print('Downloading data...')
	try:
		download_data()
	except:
		print('\t\tUnable to update 18-19 season data')

	def update_training_data():
		warnings.filterwarnings("ignore")
		
		print('Creating stats...', end='\r')
		stats_update_all = input(
			'Update all stats? \n\tY to update all stats file\n\tN to update only the last stats file: \n')
		if stats_update_all == 'Y' or stats_update_all == 'y':
			trainingData.add_stats()
		else:
			trainingData.add_stats_latest()
		print('Saving dataframe...', end='\r')
		trainingData.create_df()
		print('Done', end='\r')

	update_training_data()

