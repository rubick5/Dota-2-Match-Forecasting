import pandas as pd
import urllib.request
import urllib.parse
import json
import time
import os

# pulls everything process_raw.py needs from opendota's explorer api, which is the same
# query endpoint as the website's explorer but returns json.
# the big tables are batched by patch because the explorer times out on one large query.
# the batches are grouped to be roughly even in size rather than one file per patch,
# which would be 60 requests each.

URL = "https://api.opendota.com/api/explorer"

MATCH_COLS = [
	"match_id", "match_seq_num", "radiant_win", "start_time", "duration", "cluster",
	"human_players", "leagueid", "radiant_team_id", "dire_team_id",
	"radiant_team_complete", "dire_team_complete", "radiant_captain", "dire_captain",
	"radiant_team_name", "dire_team_name",
]
PLAYER_COLS = ["account_id", "match_id", "player_slot", "hero_id", "lane", "lane_role"]

BATCHES = {
	"6.74-6.88": ["6.74", "6.75", "6.76", "6.77", "6.78", "6.79", "6.80", "6.81", "6.82", "6.83", "6.84", "6.85", "6.86", "6.87", "6.88"],
	"7.00-7.19": ["7.00", "7.01", "7.02", "7.03", "7.04", "7.05", "7.06", "7.07", "7.08", "7.09", "7.10", "7.11", "7.12", "7.13", "7.14", "7.15", "7.16", "7.17", "7.18", "7.19"],
	"7.20-7.27": ["7.20", "7.21", "7.22", "7.23", "7.24", "7.25", "7.26", "7.27"],
	"7.28-7.31": ["7.28", "7.29", "7.30", "7.31"],
	"7.32-7.34": ["7.32", "7.33", "7.34"],
	"7.35-7.37": ["7.35", "7.36", "7.37"],
	"7.38-7.41": ["7.38", "7.39", "7.40", "7.41"],
}

def query(sql):
	# the api rejects urllib's default user agent with a 403, so set a real one
	req = urllib.request.Request(
		URL + "?" + urllib.parse.urlencode({"sql": sql}),
		headers = {"User-Agent": "dota-elo-analysis/1.0"},
	)
	for attempt in range(3):
		try:
			with urllib.request.urlopen(req, timeout = 300) as r:
				body = json.load(r)
		except Exception as e:
			print('  request failed:', e, '- retrying' if attempt < 2 else '- giving up')
			time.sleep(5)
			continue
		if body.get("err"):
			print('  query failed:', body['err'], '- retrying' if attempt < 2 else '- giving up')
			time.sleep(5)
			continue
		return body["rows"]
	raise RuntimeError("query failed three times")

def patch_list(patches):
	return ", ".join("'" + p + "'" for p in patches)

def pull_whole_table(path, sql, columns):
	if os.path.exists(path): # so an interrupted run can be resumed
		print('skipping', path, '- already pulled')
		return
	print('pulling', path)
	rows = query(sql)
	pd.DataFrame(rows, columns = columns).to_csv(path, index = False)
	print('  wrote', len(rows), 'rows')
	time.sleep(2) # the free tier is rate limited, and there is no hurry

def pull_batched(out_dir, table, columns, alias):
	# every batch is checked against a count(*) over the same join, so a truncated
	# response is caught rather than silently written out short
	os.makedirs(out_dir, exist_ok = True)
	picked = ", ".join(alias + "." + c for c in columns)
	for name, patches in BATCHES.items():
		path = out_dir + "/" + name + ".csv"
		if os.path.exists(path):
			print('skipping', path, '- already pulled')
			continue
		print('pulling', path)
		where = """
			from """ + table + " " + alias + """
			join match_patch mp on mp.match_id = """ + alias + """.match_id
			where mp.patch in (""" + patch_list(patches) + ")"
		expected = query("select count(*) as n" + where)[0]["n"]
		rows = query("select " + picked + where)
		if len(rows) != expected:
			raise RuntimeError("got " + str(len(rows)) + " rows but expected " + str(expected) + ", so the response was truncated")
		pd.DataFrame(rows, columns = columns).to_csv(path, index = False)
		print('  wrote', len(rows), 'rows')
		time.sleep(2)

os.makedirs("data/raw", exist_ok = True)
os.makedirs("data/processed", exist_ok = True)

pull_whole_table("data/raw/teams.csv", "select team_id, name, tag, logo_url from teams", ["team_id", "name", "tag", "logo_url"])
pull_whole_table("data/raw/leagues.csv", "select leagueid, ticket, banner, tier, name from leagues", ["leagueid", "ticket", "banner", "tier", "name"])

# the patch each match was played on, so hero effects can be allowed to drift.
# deriving this from start_time instead would be approximate, because consecutive
# patches overlap by a few days in the match data.
pull_whole_table("data/raw/match_patch.csv", "select match_id, patch from match_patch", ["match_id", "patch"])

pull_batched("data/raw/matches", "matches", MATCH_COLS, "m")
pull_batched("data/raw/with_heroes", "player_matches", PLAYER_COLS, "pm")

# process_raw.py reads the match batches as one csv, so stitch them together
matches_path = "data/raw/matches.csv"
if os.path.exists(matches_path):
	print('skipping', matches_path, '- already stitched')
else:
	files = sorted(os.listdir("data/raw/matches"))
	matches = pd.concat([pd.read_csv("data/raw/matches/" + f) for f in files], ignore_index = True)
	matches.to_csv(matches_path, index = False)
	print('stitched', len(matches), 'matches into', matches_path)

print('\ndone. checking what we got:')
for f in sorted(os.listdir("data/raw/with_heroes")):
	df = pd.read_csv("data/raw/with_heroes/" + f)
	print(f, len(df), 'rows,', df.match_id.nunique(), 'matches,', df.hero_id.isna().sum(), 'missing hero_id')
for f in ["data/raw/matches.csv", "data/raw/teams.csv", "data/raw/leagues.csv", "data/raw/match_patch.csv"]:
	print(f, len(pd.read_csv(f)), 'rows')
