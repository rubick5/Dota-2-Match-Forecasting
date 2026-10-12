import pandas as pd
import urllib.request
import urllib.parse
import json
import time
import os

# pulls player rows with hero_id from opendota's explorer api, which is the same query
# endpoint as the website's explorer but returns json.
# batched by patch because the explorer times out on big queries. the batches are grouped
# to be roughly even in size rather than one file per patch, which would be 60 requests.

URL = "https://api.opendota.com/api/explorer"
OUT_DIR = "data/raw/with_heroes"

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

os.makedirs(OUT_DIR, exist_ok = True)
for name, patches in BATCHES.items():
	path = OUT_DIR + "/" + name + ".csv"
	if os.path.exists(path): # so an interrupted run can be resumed
		print('skipping', name, '- already pulled')
		continue
	print('pulling', name)

	# the expected count comes from the same join, so a truncated response is caught
	# rather than silently written out short
	expected = query("""
		select count(*) as n
		from player_matches pm
		join match_patch mp on mp.match_id = pm.match_id
		where mp.patch in (""" + patch_list(patches) + ")")[0]["n"]

	rows = query("""
		select pm.account_id, pm.match_id, pm.player_slot, pm.hero_id, pm.lane, pm.lane_role
		from player_matches pm
		join match_patch mp on mp.match_id = pm.match_id
		where mp.patch in (""" + patch_list(patches) + ")")

	if len(rows) != expected:
		raise RuntimeError("got " + str(len(rows)) + " rows but expected " + str(expected) + ", so the response was truncated")

	df = pd.DataFrame(rows, columns = ["account_id", "match_id", "player_slot", "hero_id", "lane", "lane_role"])
	df.to_csv(path, index = False)
	print('  wrote', len(df), 'rows to', path)
	time.sleep(2) # the free tier is rate limited, and there is no hurry

# the patch each match was played on, so hero effects can be allowed to drift.
# deriving this from start_time instead would be approximate, because consecutive
# patches overlap by a few days in the match data.
patch_path = "data/raw/match_patch.csv"
if os.path.exists(patch_path):
	print('skipping match_patch - already pulled')
else:
	print('pulling match_patch')
	rows = query("select match_id, patch from match_patch")
	pd.DataFrame(rows, columns = ["match_id", "patch"]).to_csv(patch_path, index = False)
	print('  wrote', len(rows), 'rows to', patch_path)

print('\ndone. checking what we got:')
files = sorted(os.listdir(OUT_DIR))
total = 0
for f in files:
	df = pd.read_csv(OUT_DIR + "/" + f)
	total += len(df)
	print(f, len(df), 'rows,', df.match_id.nunique(), 'matches,', df.hero_id.isna().sum(), 'missing hero_id')
print('total', total, 'rows')
