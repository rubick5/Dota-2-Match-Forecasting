# exploratory. superseded by evaluate.py in the repo root, which is where the readme's
# numbers come from.
# this scores every match it trained on and takes its baseline from that same data, so
# nothing printed here is a held out result and none of it is comparable to the readme.
# kept as a record of how the model developed. run it from the repo root, not this folder.

import pandas as pd
import math
import numpy as np

data = pd.read_parquet('data/processed/data.parquet')
"""
<class 'pandas.DataFrame'>
RangeIndex: 210238 entries, 0 to 210237
Data columns (total 20 columns):
 #   Column                 Non-Null Count   Dtype  
---  ------                 --------------   -----  
 0   match_id               210238 non-null  Int64  
 1   match_seq_num          210238 non-null  Int64  
 2   radiant_win            210238 non-null  boolean
 3   start_time             210238 non-null  Int64  
 4   duration               210238 non-null  Int64  
 5   cluster                210238 non-null  Int64  
 6   human_players          210238 non-null  Int64  
 7   leagueid               210238 non-null  Int64  
 8   radiant_team_id        210238 non-null  Int64  
 9   dire_team_id           210238 non-null  Int64  
 10  radiant_team_complete  210238 non-null  Int64  
 11  dire_team_complete     210238 non-null  Int64  
 12  radiant_captain        196636 non-null  Int64  
 13  dire_captain           196638 non-null  Int64  
 14  radiant_team_name      207435 non-null  object 
 15  dire_team_name         207468 non-null  object 
 16  ticket                 5763 non-null    object 
 17  banner                 5763 non-null    object 
 18  tier                   210238 non-null  object 
 19  league_name            210238 non-null  object 
dtypes: Int64(13), boolean(1), object(6)
memory usage: 33.5+ MB
None
"""
for K in [8, 16, 32, 64, 128]:
	print('Doing everything with K =', K)
	class TeamTracker:
		def __init__(self, name):
			self.__next_match = 1
			self.__elo = 1500
			self.__name = name
		def get_name(self):
			return self.__name

		def get_elo(self) -> int:
			return self.__elo
		
		def get_matches(self) -> int:
			return self.__next_match

		def calc_er(self, other_elo):
			return 1 / (1 + 10 ** ((other_elo - self.__elo) / 400))

		def played_match(self, won: bool, other_elo: int):
			er = self.calc_er(other_elo)
			self.__elo += K * ((1 if won else 0) - er)
			self.__next_match += 1


	team_trackers = {}
	records = []
	def log_loss(p, y):
		return -(y * np.log(p) + (1 - y) * np.log(1 - p))

	data = data[~data['radiant_captain'].isna() & ~data['dire_captain'].isna()].sort_values('start_time')
	print(len(data))
	for row in data.itertuples():
		if row.radiant_captain not in team_trackers:
			team_trackers[row.radiant_captain] = TeamTracker(row.radiant_captain)

		if row.dire_captain not in team_trackers:
			team_trackers[row.dire_captain] = TeamTracker(row.dire_captain)

		radiant_elo = team_trackers[row.radiant_captain].get_elo()
		dire_elo = team_trackers[row.dire_captain].get_elo()

		p = team_trackers[row.radiant_captain].calc_er(dire_elo)
		y = 1 if row.radiant_win else 0
		records.append({
			"match_id":   row.match_id,
			"start_time": row.start_time,
			"tier":       row.tier,
			"p":          p, # what we predicted
			"y":          y, # what happened
			"r_games":    team_trackers[row.radiant_captain].get_matches(), # radiant games
			"d_games":    team_trackers[row.dire_captain].get_matches(), # dire games
		})

		team_trackers[row.radiant_captain].played_match(row.radiant_win, dire_elo)
		team_trackers[row.dire_captain].played_match(not row.radiant_win, radiant_elo)


	ratings = pd.DataFrame(
		[(tid, t.get_name(), t.get_elo(), t.get_matches()) for tid, t in team_trackers.items()],
		columns=["team_id", "team_name", "elo", "matches"],
	)

	# records: let's see how well we did:
	res = pd.DataFrame(records, columns=["match_id", "start_time", "tier", "p", "y", "r_games", "d_games"])
	#print(res.info())
	res_proper = res[(res['r_games'] > 10) & (res['d_games'] > 10)]
	res_proper['p'] = res.p.clip(1e-15, 1 - 1e-15)
	#print("res.p.max():", res.p.max(), "res.p.min:", res.p.min())
	#print("elo described:\n", ratings.elo.describe())
	print("avg log loss:", log_loss(res_proper['p'], res_proper['y']).mean())

	base_radiant_win_rate = res_proper.y.mean()
	print("max_loss", log_loss(base_radiant_win_rate, res_proper['y']).mean())