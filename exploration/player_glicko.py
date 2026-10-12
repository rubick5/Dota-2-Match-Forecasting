# exploratory. superseded by evaluate.py in the repo root, which is where the readme's
# numbers come from.
# same caveats as player_elo.py: C is picked by eye across runs that include the held out
# loss, and the baseline is measured on the training data.
# the one thing worth keeping here is the loss by time decile, which is what showed the
# model getting worse over time and the cold start games being easier than average.
# run it from the repo root, not this folder.

import pandas as pd
import math
import numpy as np

data = pd.read_parquet('data/processed/data.parquet')
cutoff = data['start_time'].quantile(0.9)
test_data = data[data.start_time > cutoff]
data = data[data.start_time <= cutoff]
player_matches = pd.read_parquet('data/processed/player_matches.parquet')

Q = math.log(10) / 400
START_RD = 350 # new players are very uncertain, so their first few games move them a lot
MIN_RD = 50 # stops established players from freezing
DAY = 60 * 60 * 24

class PlayerTracker:
		def __init__(self, account_id, start_time):
			self.__matches = 0
			self.__elo = 1500
			self.__rd = START_RD
			self.__last_played = start_time
			self.__account_id = account_id

		def get_id(self):
			return self.__account_id

		def get_elo(self) -> float:
			return self.__elo

		def get_rd(self) -> float:
			return self.__rd

		def get_matches(self) -> int:
			return self.__matches

		def decay(self, now, c):
			# uncertainty creeps back up the longer a player is inactive
			days = (now - self.__last_played) / DAY
			self.__rd = min(math.sqrt(self.__rd ** 2 + c ** 2 * days), START_RD)

		def played_match(self, now, won: bool, other_elo, other_rd):
			g = calc_g(other_rd)
			er = calc_er(self.__elo, other_elo, other_rd)
			d_sq = 1 / (Q ** 2 * g ** 2 * er * (1 - er))
			new_rd = 1 / math.sqrt(1 / self.__rd ** 2 + 1 / d_sq)
			self.__elo += Q * new_rd ** 2 * g * ((1 if won else 0) - er)
			self.__rd = max(new_rd, MIN_RD)
			self.__last_played = now
			self.__matches += 1


def calc_g(rd):
	return 1 / math.sqrt(1 + 3 * Q ** 2 * rd ** 2 / math.pi ** 2)

def calc_er(elo, other_elo, other_rd):
	return 1 / (1 + 10 ** (-calc_g(other_rd) * (elo - other_elo) / 400))

def log_loss(p, y):
	return -(y * np.log(p) + (1 - y) * np.log(1 - p))

rosters = (player_matches
	.assign(is_radiant = player_matches.player_slot < 128)
	.groupby(['match_id', 'is_radiant'])
	.account_id.apply(list)
	.to_dict()
)

def team_rating(trackers, team):
	elo = sum(trackers[p].get_elo() for p in team) / len(team)
	rd = math.sqrt(sum(trackers[p].get_rd() ** 2 for p in team) / len(team))
	return elo, rd

for C in [0, 5, 10, 20, 40]: # change numbers if you want to tune
	print('Doing everything with C =', C)

	player_trackers = {}
	records = []

	data = data.sort_values('start_time')
	for row in data.itertuples():
		radiant = rosters[(row.match_id, True)]
		dire = rosters[(row.match_id, False)]
		for p in radiant + dire:
			if p not in player_trackers:
				player_trackers[p] = PlayerTracker(p, row.start_time)
			player_trackers[p].decay(row.start_time, C)

		radiant_elo, radiant_rd = team_rating(player_trackers, radiant)
		dire_elo, dire_rd = team_rating(player_trackers, dire)
		prob = calc_er(radiant_elo, dire_elo, math.sqrt(radiant_rd ** 2 + dire_rd ** 2)) # what we predicted
		y = 1 if row.radiant_win else 0

		records.append({
			"match_id":   row.match_id,
			"start_time": row.start_time,
			"tier":       row.tier,
			"p":          prob, # what we predicted
			"y":          y, # what happened
			"r_games":    sum(player_trackers[p].get_matches() for p in radiant) / len(radiant), # radiant games
			"d_games":    sum(player_trackers[p].get_matches() for p in dire) / len(dire), # dire games
		})

		for p in radiant:
			player_trackers[p].played_match(row.start_time, row.radiant_win, dire_elo, dire_rd)

		for p in dire:
			player_trackers[p].played_match(row.start_time, not row.radiant_win, radiant_elo, radiant_rd)

	ratings = pd.DataFrame(
		[(tid, t.get_elo(), t.get_rd(), t.get_matches()) for tid, t in player_trackers.items()],
		columns=["player_id", "elo", "rd", "matches"],
	)

	# records: let's see how well we did:
	res = pd.DataFrame(records, columns=["match_id", "start_time", "tier", "p", "y", "r_games", "d_games"])
	res_proper = res[(res['r_games'] > 10) & (res['d_games'] > 10)].copy()
	res_proper['p'] = res_proper.p.clip(1e-15, 1 - 1e-15)
	print("avg log loss:", log_loss(res_proper['p'], res_proper['y']).mean())

	base_radiant_win_rate = res_proper.y.mean()
	print("max_loss", log_loss(base_radiant_win_rate, res_proper['y']).mean())

	# also look at the cold-start games, which is what this file is for:
	res_new = res[(res['r_games'] <= 10) | (res['d_games'] <= 10)].copy()
	res_new['p'] = res_new.p.clip(1e-15, 1 - 1e-15)
	print("avg log loss on cold-start games:", log_loss(res_new['p'], res_new['y']).mean())

	# does the model get worse as time goes on?
	res['bucket'] = pd.qcut(res['start_time'], 10, labels = False)
	res['loss'] = log_loss(res.p.clip(1e-15, 1 - 1e-15), res.y)
	res['cold'] = (res['r_games'] <= 10) | (res['d_games'] <= 10)
	print("loss by time decile:")
	print(res.groupby('bucket').agg(loss = ('loss', 'mean'), cold_share = ('cold', 'mean')))
	print("Described ratings:")
	print(ratings.describe())
	print("\n\n\n")

	test_records = []
	# time to test on the witheld data:
	for row in test_data.itertuples():
		radiant = rosters[(row.match_id, True)]
		dire = rosters[(row.match_id, False)]
		for p in radiant + dire:
			if p not in player_trackers:
				player_trackers[p] = PlayerTracker(p, row.start_time)
			player_trackers[p].decay(row.start_time, C)

		radiant_elo, radiant_rd = team_rating(player_trackers, radiant)
		dire_elo, dire_rd = team_rating(player_trackers, dire)
		prob = calc_er(radiant_elo, dire_elo, math.sqrt(radiant_rd ** 2 + dire_rd ** 2))
		y = 1 if row.radiant_win else 0

		test_records.append({
			"match_id":   row.match_id,
			"start_time": row.start_time,
			"tier":       row.tier,
			"p":          prob, # what we predicted
			"y":          y, # what happened
			"r_games":    sum(player_trackers[p].get_matches() for p in radiant) / len(radiant), # radiant games
			"d_games":    sum(player_trackers[p].get_matches() for p in dire) / len(dire), # dire games
		})
	test_results = pd.DataFrame(test_records, columns=["match_id", "start_time", "tier", "p", "y", "r_games", "d_games"])
	test_results = test_results[(test_results['r_games'] > 10) & (test_results['d_games'] > 10)].copy()
	test_results['p'] = test_results.p.clip(1e-15, 1 - 1e-15)
	print('loss on held out data:')
	print("avg log loss:", log_loss(test_results['p'], test_results['y']).mean())
