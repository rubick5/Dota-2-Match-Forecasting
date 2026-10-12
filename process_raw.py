import pandas as pd
import glob
matches = pd.read_csv("data/raw/matches.csv")
teams = pd.read_csv("data/raw/teams.csv")
leagues = pd.read_csv("data/raw/leagues.csv")
"""
print(matches.shape)
print(matches.head())
print(matches.dtypes)
print(matches.info()) """

# similar to df |> filter(is_na(radiant_win))
#print(matches[matches.radiant_win.isna()])

df = matches[matches.radiant_win.notna() & matches.radiant_team_id.notna() & matches.dire_team_id.notna()]
df = df.infer_objects()

df = df.drop(['radiant_team_name', 'dire_team_name'], axis = 1)
"""print(df.shape)
print(df.dtypes)
print(df.info()) """
#print(df['radiant_captain'].value_counts())

with_rad_teams = df.merge(
	teams[['team_id', 'name']].rename(columns = {'team_id': 'radiant_team_id', 'name': 'radiant_team_name'}),
	on = 'radiant_team_id',
	how = 'left',
	validate = "m:1",
)

with_teams = with_rad_teams.merge(
	teams[['team_id', 'name']].rename(columns = {'team_id': 'dire_team_id', 'name': 'dire_team_name'}),
	on = 'dire_team_id',
	how = 'left',
	validate = "m:1",

)

with_leagues = with_teams.merge(
	leagues.rename(columns = {'name': 'league_name'}),
	on = 'leagueid',
	how = 'left',
	validate = 'm:1'
).convert_dtypes()



model_df = with_leagues[with_leagues.tier.isin(["premium", "professional"])].sort_values("start_time").reset_index(drop = True)

players_pattern = "data/raw/with_heroes/*.csv" # was data/raw/players/*.csv, which had no hero_id
files = glob.glob(players_pattern)
player_matches = pd.concat([pd.read_csv(f, dtype={"account_id" : "string"}) for f in files], ignore_index=True)

# need to check that all matches have 10 players:
counts = player_matches.groupby('match_id').size()
bad = counts[counts != 10]
print('number of games with incomplete players:', len(bad))
player_matches = player_matches[~player_matches['match_id'].isin(bad.index)]

#player_matches.info(show_counts = True)
model_df = model_df[model_df['match_id'].isin(player_matches['match_id'])]

# the patch is needed to let hero effects drift, since a hero id means something
# different in 6.74 than it does in 7.41
match_patch = pd.read_csv("data/raw/match_patch.csv", dtype = {"patch": "string"}) # else 7.40 is read as the float 7.4
model_df = model_df.merge(
	match_patch,
	on = 'match_id',
	how = 'left',
	validate = 'm:1',
)
print('matches with no patch:', model_df.patch.isna().sum())
player_matches.to_parquet('data/processed/player_matches.parquet')
model_df.to_parquet('data/processed/data.parquet')