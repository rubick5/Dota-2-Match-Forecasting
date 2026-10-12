import pandas as pd
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, hstack

# one pipeline for every model, so the comparison table is held out and like for like.
# every model is run once, front to back, and only ever predicts a match before learning from it,
# so one pass gives honest predictions for the train, validation and test periods at once.

MIN_GAMES = 10 # a side is only scored once its players average more than this many earlier games
EDGES = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # time quantiles, 7 chunks. the last chunk is the old 0.9 held out cutoff
START_RD = 350 # new players are very uncertain, so their first few games move them a lot
MIN_RD = 50 # stops established players from freezing
DAY = 60 * 60 * 24
Q = math.log(10) / 400

data = pd.read_parquet('data/processed/data.parquet').sort_values(['start_time', 'match_id']).reset_index(drop = True)
player_matches = pd.read_parquet('data/processed/player_matches.parquet')
n = len(data)
y = data.radiant_win.astype('int64').to_numpy()
start_times = data.start_time.astype('float64').to_numpy()
chunk = np.digitize(start_times, data.start_time.quantile(EDGES).to_numpy())

rosters = (player_matches
	.assign(is_radiant = player_matches.player_slot < 128)
	.groupby(['match_id', 'is_radiant'])
	.account_id.apply(list)
	.to_dict()
)
radiant_rosters = [rosters[(m, True)] for m in data.match_id]
dire_rosters = [rosters[(m, False)] for m in data.match_id]

# how many matches each side's players had played before this one. nothing to do with any model,
# so the same matches get scored for all of them.
games = {}
r_games = np.zeros(n)
d_games = np.zeros(n)
for i in range(n):
	radiant = radiant_rosters[i]
	dire = dire_rosters[i]
	r_games[i] = sum(games.get(p, 0) for p in radiant) / len(radiant)
	d_games[i] = sum(games.get(p, 0) for p in dire) / len(dire)
	for p in radiant + dire:
		games[p] = games.get(p, 0) + 1
eligible = (r_games > MIN_GAMES) & (d_games > MIN_GAMES)
print('matches:', n, 'scored once burnt in:', eligible.sum())


def calc_g(rd):
	return 1 / math.sqrt(1 + 3 * Q ** 2 * rd ** 2 / math.pi ** 2)

def calc_er(elo, other_elo, other_rd = 0): # other_rd = 0 is plain elo
	return 1 / (1 + 10 ** (-calc_g(other_rd) * (elo - other_elo) / 400))

def log_loss(p, y):
	p = np.clip(p, 1e-15, 1 - 1e-15)
	return -(y * np.log(p) + (1 - y) * np.log(1 - p))


class Tracker:
		def __init__(self):
			self.__elo = 1500

		def get_elo(self) -> float:
			return self.__elo

		def played_match(self, elo_change):
			self.__elo += elo_change


class GlickoTracker:
		def __init__(self, start_time):
			self.__elo = 1500
			self.__rd = START_RD
			self.__last_played = start_time

		def get_elo(self) -> float:
			return self.__elo

		def get_rd(self) -> float:
			return self.__rd

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


def run_keyed_elo(K, radiant_keys, dire_keys):
	trackers = {}
	p = np.full(n, 0.5)
	for i in range(n):
		r = radiant_keys[i]
		d = dire_keys[i]
		if pd.isna(r) or pd.isna(d): # no captain recorded, so nothing to rate. predicts 0.5
			continue
		for k in (r, d):
			if k not in trackers:
				trackers[k] = Tracker()
		p[i] = calc_er(trackers[r].get_elo(), trackers[d].get_elo())
		elo_change = K * (y[i] - p[i])
		trackers[r].played_match(elo_change)
		trackers[d].played_match(-elo_change)
	return p

def run_roster_elo(K):
	trackers = {}
	p = np.zeros(n)
	for i in range(n):
		radiant = radiant_rosters[i]
		dire = dire_rosters[i]
		for q in radiant + dire:
			trackers.setdefault(q, Tracker())
		radiant_elo = sum(trackers[q].get_elo() for q in radiant) / len(radiant)
		dire_elo = sum(trackers[q].get_elo() for q in dire) / len(dire)
		p[i] = calc_er(radiant_elo, dire_elo)
		elo_change = K * (y[i] - p[i])
		for q in radiant:
			trackers[q].played_match(elo_change)
		for q in dire:
			trackers[q].played_match(-elo_change)
	return p

def run_roster_glicko(C):
	trackers = {}
	p = np.zeros(n)
	for i in range(n):
		radiant = radiant_rosters[i]
		dire = dire_rosters[i]
		now = start_times[i]
		for q in radiant + dire:
			if q not in trackers:
				trackers[q] = GlickoTracker(now)
			trackers[q].decay(now, C)
		radiant_elo = sum(trackers[q].get_elo() for q in radiant) / len(radiant)
		dire_elo = sum(trackers[q].get_elo() for q in dire) / len(dire)
		radiant_rd = math.sqrt(sum(trackers[q].get_rd() ** 2 for q in radiant) / len(radiant))
		dire_rd = math.sqrt(sum(trackers[q].get_rd() ** 2 for q in dire) / len(dire))
		p[i] = calc_er(radiant_elo, dire_elo, math.sqrt(radiant_rd ** 2 + dire_rd ** 2))
		for q in radiant:
			trackers[q].played_match(now, y[i] == 1, dire_elo, dire_rd)
		for q in dire:
			trackers[q].played_match(now, y[i] == 0, radiant_elo, radiant_rd)
	return p


team_radiant = data.radiant_team_id.tolist()
team_dire = data.dire_team_id.tolist()
captain_radiant = data.radiant_captain.tolist()
captain_dire = data.dire_captain.tolist()

K_GRID = [8, 16, 32, 64, 128] # change numbers if you want to tune
C_GRID = [0, 5, 10, 20]
MODELS = {
	'team':    (K_GRID, lambda K: run_keyed_elo(K, team_radiant, team_dire)),
	'captain': (K_GRID, lambda K: run_keyed_elo(K, captain_radiant, captain_dire)),
	'roster':  (K_GRID, run_roster_elo),
	'glicko':  (C_GRID, run_roster_glicko),
}

preds = {}
for name, (grid, run) in MODELS.items():
	for param in grid:
		print('running', name, 'with', param)
		preds[(name, param)] = run(param)


# walk forward: for each fold the test chunk is scored once, with the parameter that did best on the chunk before it.
folds = list(range(2, len(EDGES) + 1))
HEADLINE = folds[-1]
selected = {}
rows = []
for fold in folds:
	val = eligible & (chunk == fold - 1)
	test = eligible & (chunk == fold)
	base_rate = y[eligible & (chunk < fold)].mean() # the constant guess, from before the test period
	base_loss = log_loss(base_rate, y[test]).mean()
	for name, (grid, run) in MODELS.items():
		val_losses = {param: log_loss(preds[(name, param)][val], y[val]).mean() for param in grid}
		best = min(val_losses, key = val_losses.get)
		selected[(fold, name)] = best
		test_loss = log_loss(preds[(name, best)][test], y[test]).mean()
		rows.append({
			"fold":      fold,
			"model":     name,
			"param":     best,
			"val_loss":  val_losses[best],
			"test_loss": test_loss,
			"baseline":  base_loss,
			"n_test":    test.sum(),
		})
results = pd.DataFrame(rows)
print('\nheld out results, one row per fold (param chosen on the chunk before the test chunk):')
print(results.to_string(index = False))

print('\nwalk forward summary over', len(folds), 'folds:')
results['gain'] = results.baseline - results.test_loss
print(results.groupby('model').agg(
	test_loss_mean = ('test_loss', 'mean'),
	test_loss_std  = ('test_loss', 'std'),
	gain_mean      = ('gain', 'mean'),
	gain_std       = ('gain', 'std'),
))

headline = results[results.fold == HEADLINE].set_index('model')
test = eligible & (chunk == HEADLINE)
start = pd.to_datetime(start_times[test].min(), unit = 's')
end = pd.to_datetime(start_times[test].max(), unit = 's')
print('\nheadline: scored once on', test.sum(), 'matches from', start.date(), 'to', end.date())
print('baseline (constant guess from before the test period):', headline.baseline.iloc[0])
print('baseline if you cheat and use the test win rate:', log_loss(y[test].mean(), y[test]).mean())
base_loss = log_loss(y[eligible & (chunk < HEADLINE)].mean(), y[test])
for name in MODELS:
	model_loss = log_loss(preds[(name, selected[(HEADLINE, name)])][test], y[test])
	diff = base_loss - model_loss # per match, so we can see how much of the gain is noise
	print(name, 'param', selected[(HEADLINE, name)], 'log loss', model_loss.mean(),
		'gain over baseline', diff.mean(), '+-', diff.std() / math.sqrt(len(diff)))
roster_loss = log_loss(preds[('roster', selected[(HEADLINE, 'roster')])][test], y[test])
for name in ['team', 'captain', 'glicko']:
	diff = log_loss(preds[(name, selected[(HEADLINE, name)])][test], y[test]) - roster_loss
	print(name, 'minus roster:', diff.mean(), '+-', diff.std() / math.sqrt(len(diff)))


# calibration on the held out chunk: do the probabilities mean what they say?
def calibration(p, y, bins = 10):
	edges = np.linspace(0, 1, bins + 1)
	which = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
	ece = 0
	for b in range(bins):
		in_bin = which == b
		if in_bin.any():
			ece += in_bin.mean() * abs(p[in_bin].mean() - y[in_bin].mean())
	return ece

print('\ncalibration on the headline chunk:')
fig, (ax_cal, ax_hist) = plt.subplots(1, 2, figsize = (12, 5))
ax_cal.plot([0, 1], [0, 1], color = 'grey', linestyle = '--')
for name in MODELS:
	p = preds[(name, selected[(HEADLINE, name)])][test]
	y_test = y[test]
	print(name,
		'ECE', round(calibration(p, y_test), 4),
		'| std of p', round(p.std(), 4),
		'| 5th/95th pct', np.round(np.percentile(p, [5, 95]), 3),
		'| share of p in 0.45-0.55', round(((p > 0.45) & (p < 0.55)).mean(), 3),
	)
	# equal count bins, because the predictions bunch up near the middle
	order = np.argsort(p)
	groups = np.array_split(order, 10)
	ax_cal.plot([p[g].mean() for g in groups], [y_test[g].mean() for g in groups], marker = 'o', label = name)
	ax_hist.hist(p, bins = 40, range = (0, 1), histtype = 'step', label = name)
ax_cal.set_xlabel('predicted radiant win probability')
ax_cal.set_ylabel('observed radiant win rate')
ax_cal.set_title('reliability (10 equal count bins)')
ax_cal.legend()
ax_hist.set_xlabel('predicted radiant win probability')
ax_hist.set_ylabel('matches')
ax_hist.set_title('spread of predictions')
ax_hist.legend()
fig.tight_layout()
fig.savefig('calibration.png', dpi = 150)
print('saved calibration.png')


# does the draft add anything on top of the rating difference?
def fit_logistic(X, y_, l2, penalised, weights = None):
	w = np.ones(X.shape[0]) if weights is None else weights
	w = w / w.sum() # so the l2 scale means the same thing whatever the weights are
	def loss_and_grad(b):
		z = X @ b
		loss = (w * (np.logaddexp(0, z) - y_ * z)).sum() + l2 / 2 * (penalised * b ** 2).sum()
		grad = np.asarray(X.T @ (w * (1 / (1 + np.exp(-z)) - y_))).ravel() + l2 * penalised * b
		return loss, grad
	return minimize(loss_and_grad, np.zeros(X.shape[1]), jac = True, method = 'L-BFGS-B').x

def predict_logistic(X, w):
	return 1 / (1 + np.exp(-(X @ w)))

print('\ndraft ablation:')
if 'hero_id' not in player_matches.columns:
	print('skipped: data/processed/player_matches.parquet has no hero_id column.',
		'add hero_id (per player per match) in process_raw.py and rerun.')
else:
	# a hero id means something different in 6.74 than in 7.41, so the hero weights get two
	# chances to drift: by weighting recent matches more, and by letting each era deviate
	# from a shared global weight.
	ERAS = { # the same grouping as the fetch batches, which were sized to be roughly even
		"6.74-6.88": ["6.74", "6.75", "6.76", "6.77", "6.78", "6.79", "6.80", "6.81", "6.82", "6.83", "6.84", "6.85", "6.86", "6.87", "6.88"],
		"7.00-7.19": ["7.00", "7.01", "7.02", "7.03", "7.04", "7.05", "7.06", "7.07", "7.08", "7.09", "7.10", "7.11", "7.12", "7.13", "7.14", "7.15", "7.16", "7.17", "7.18", "7.19"],
		"7.20-7.27": ["7.20", "7.21", "7.22", "7.23", "7.24", "7.25", "7.26", "7.27"],
		"7.28-7.31": ["7.28", "7.29", "7.30", "7.31"],
		"7.32-7.34": ["7.32", "7.33", "7.34"],
		"7.35-7.37": ["7.35", "7.36", "7.37"],
		"7.38-7.41": ["7.38", "7.39", "7.40", "7.41"],
	}
	HALF_LIVES = [None, 1460, 730, 365] # days. None means every match counts the same
	L2_GRID = [0.001, 0.003, 0.01, 0.03, 0.1]
	L2_ERA_GRID = [0.03, 0.1, 0.3] # era deviations are shrunk harder than the global weights

	era_of = {p: i for i, (era, patches) in enumerate(ERAS.items()) for p in patches}
	era_index = data.patch.map(era_of)
	print('matches with no era:', era_index.isna().sum())
	era_index = era_index.fillna(0).astype(int).to_numpy()

	pm = player_matches[player_matches.hero_id.notna()]
	match_row = pd.Series(np.arange(n), index = data.match_id.astype('int64'))
	row = pm.match_id.map(match_row)
	known = row.notna()
	rows = row[known].astype(int).to_numpy()
	sign = np.where(pm.player_slot[known] < 128, 1, -1) # +1 for a radiant pick, -1 for a dire pick
	heroes = sorted(pm.hero_id.unique())
	hero_col = {h: j for j, h in enumerate(heroes)}
	cols = pm.hero_id[known].map(hero_col).to_numpy()
	H = csr_matrix((sign, (rows, cols)), shape = (n, len(heroes)))
	# one block of hero columns per era, and a match only ever fills its own block
	H_era = csr_matrix((sign, (rows, era_index[rows] * len(heroes) + cols)), shape = (n, len(ERAS) * len(heroes)))

	ones = np.ones((n, 1))
	ablation = []
	headline_losses = {}
	for fold in folds:
		train = eligible & (chunk < fold - 1)
		val = eligible & (chunk == fold - 1)
		test = eligible & (chunk == fold)
		p_elo = preds[('roster', selected[(fold, 'roster')])]
		rating = np.log(p_elo / (1 - p_elo)).reshape(-1, 1) # the rating difference, on the logit scale
		X_rating = csr_matrix(np.hstack([ones, rating]))
		X_draft = hstack([X_rating, H]).tocsr()
		X_era = hstack([X_rating, H, H_era]).tocsr()
		pen_rating = np.zeros(2) # the intercept and the rating itself are never shrunk
		pen_draft = np.concatenate([pen_rating, np.ones(len(heroes))])

		# recency weights are measured back from the last match the model is allowed to see
		t_ref = start_times[train | val].max()
		def decay_weights(half_life, mask):
			if half_life is None:
				return None
			return 0.5 ** ((t_ref - start_times[mask]) / (half_life * DAY))

		row_loss = {"fold": fold, "elo": log_loss(p_elo[test], y[test]).mean()}
		variants = [
			('rating',       X_rating, pen_rating, [(0, None, None)]),
			('rating+draft', X_draft,  pen_draft,  [(l2, None, None) for l2 in L2_GRID]),
			('draft+decay',  X_draft,  pen_draft,  [(l2, None, hl) for l2 in L2_GRID for hl in HALF_LIVES]),
			('draft+era',    X_era,    None,       [(l2, l2e, None) for l2 in L2_GRID for l2e in L2_ERA_GRID]),
		]
		for label, X, pen, grid in variants:
			val_losses = {}
			for l2, l2_era, half_life in grid:
				p = pen if pen is not None else np.concatenate([pen_draft, np.full(len(ERAS) * len(heroes), l2_era / l2)])
				b = fit_logistic(X[train], y[train], l2, p, decay_weights(half_life, train))
				val_losses[(l2, l2_era, half_life)] = log_loss(predict_logistic(X[val], b), y[val]).mean()
			l2, l2_era, half_life = min(val_losses, key = val_losses.get)
			p = pen if pen is not None else np.concatenate([pen_draft, np.full(len(ERAS) * len(heroes), l2_era / l2)])
			b = fit_logistic(X[train | val], y[train | val], l2, p, decay_weights(half_life, train | val))
			row_loss[label] = log_loss(predict_logistic(X[test], b), y[test]).mean()
			if label == 'draft+decay':
				row_loss['half_life'] = half_life
			if fold == HEADLINE:
				headline_losses[label] = log_loss(predict_logistic(X[test], b), y[test])
		ablation.append(row_loss)
	ablation = pd.DataFrame(ablation)
	print(ablation.to_string(index = False))
	print(ablation[['elo', 'rating', 'rating+draft', 'draft+decay', 'draft+era']].agg(['mean', 'std']))

	print('\nheadline chunk, paired differences (positive means the second one is better):')
	for a, b in [('rating', 'rating+draft'), ('rating+draft', 'draft+decay'), ('rating+draft', 'draft+era'), ('draft+decay', 'draft+era')]:
		diff = headline_losses[a] - headline_losses[b]
		print(f'  {a} -> {b}:', diff.mean(), '+-', diff.std() / math.sqrt(len(diff)))
