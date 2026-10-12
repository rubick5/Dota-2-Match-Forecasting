# exploratory. superseded by evaluate.py in the repo root, which is where the readme's
# numbers come from.
# this does hold out the last 10%, but K is picked by eye across runs that include the
# held out loss, and the baseline it compares against is measured on the training data,
# so the comparison is optimistic. the burn in filter here also counts the current match.
# kept as a record of how the model developed. run it from the repo root, not this folder.

import pandas as pd
import math
import numpy as np

data = pd.read_parquet('data/processed/data.parquet')
cutoff = data['start_time'].quantile(0.9)
test_data = data[data.start_time > cutoff]
data = data[data.start_time <= cutoff]
player_matches = pd.read_parquet('data/processed/player_matches.parquet')
class PlayerTracker:
		def __init__(self, account_id):
			self.__matches = 0
			self.__elo = 1500
			self.__account_id = account_id

		def get_id(self):
			return self.__account_id

		def get_elo(self) -> int:
			return self.__elo
		
		def get_matches(self) -> int:
			return self.__matches

		
		def played_match(self, elo_change):
			self.__elo += elo_change
			self.__matches += 1

			
for K in [8, 16, 32, 64, 128]: # change numbers if you want to tune
	print('Doing everything with K =', K)
	def calc_er(elo, other_elo):
		return 1 / (1 + 10 ** ((other_elo - elo) / 400))
	
	def elo_change(won: bool, elo, other_elo):
		er = calc_er(elo, other_elo)
		return K * ((1 if won else 0) - er)

	player_trackers = {}
	records = []

	def log_loss(p, y):
		return -(y * np.log(p) + (1 - y) * np.log(1 - p))

	data = data.sort_values('start_time')
	rosters = (player_matches
		.assign(is_radiant = player_matches.player_slot < 128)
		.groupby(['match_id', 'is_radiant'])
		.account_id.apply(list)
		.to_dict()
	)
	#print(len(data))
	for row in data.itertuples():
		radiant = rosters[(row.match_id, True)]
		dire = rosters[(row.match_id, False)]
		for p in radiant + dire: # idk if this works
			if p not in player_trackers:
				player_trackers[p] = PlayerTracker(p)

		radiant_elo = sum(player_trackers[p].get_elo()for p in radiant) / len(radiant)
		dire_elo = sum(player_trackers[p].get_elo()for p in dire) / len(dire)
		prob = calc_er (radiant_elo, dire_elo)
		y = 1 if row.radiant_win else 0


		radiant_elo_change = elo_change(row.radiant_win, radiant_elo, dire_elo)
		dire_elo_change = elo_change(not row.radiant_win, dire_elo, radiant_elo)
		for p in radiant:
			player_trackers[p].played_match(radiant_elo_change)

		for p in dire:
			player_trackers[p].played_match(dire_elo_change)

		records.append({
			"match_id":   row.match_id,
			"start_time": row.start_time,
			"tier":       row.tier,
			"p":          prob, # what we predicted
			"y":          y, # what happened
			"r_games":    sum(player_trackers[p].get_matches() for p in radiant) / len(radiant), # radiant games
			"d_games":    sum(player_trackers[p].get_matches() for p in dire) / len(dire), # radiant games
		})

	ratings = pd.DataFrame(
		[(tid, t.get_elo(), t.get_matches()) for tid, t in player_trackers.items()],
		columns=["player_id", "elo", "matches"],
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
	print("Described ratings:")
	print(ratings.describe())
	print("\n\n\n")
	test_records = []
	# time to test on the witheld data:
	for row in test_data.itertuples():
		radiant = rosters[(row.match_id, True)]
		dire = rosters[(row.match_id, False)]
		for p in radiant + dire: # idk if this works
			if p not in player_trackers:
				player_trackers[p] = PlayerTracker(p)

		radiant_elo = sum(player_trackers[p].get_elo()for p in radiant) / len(radiant)
		dire_elo = sum(player_trackers[p].get_elo()for p in dire) / len(dire)
		prob = calc_er (radiant_elo, dire_elo)
		y = 1 if row.radiant_win else 0

		test_records.append({
			"match_id":   row.match_id,
			"start_time": row.start_time,
			"tier":       row.tier,
			"p":          prob, # what we predicted
			"y":          y, # what happened
			"r_games":    sum(player_trackers[p].get_matches() for p in radiant) / len(radiant), # radiant games
			"d_games":    sum(player_trackers[p].get_matches() for p in dire) / len(dire), # radiant games
		})
	test_results = pd.DataFrame(test_records, columns=["match_id", "start_time", "tier", "p", "y", "r_games", "d_games"])
	test_results = test_results[(test_results['r_games'] > 10) & (test_results['d_games'] > 10)]
	test_results['p'] = test_results.p.clip(1e-15, 1 - 1e-15)
	print('loss on held out data:')
	print("avg log loss:", log_loss(test_results['p'], test_results['y']).mean())