# scratch work, not part of the pipeline. this is where the churn problem was first
# measured: how many teams in the test period were never seen in training, and what
# share of test matches involve one. that is what motivated moving the rating off the
# team id and onto the players.
# run it from the repo root, not this folder.

import pandas as pd
import glob

data = pd.read_parquet('data/processed/data.parquet')
player_matches = pd.read_parquet('data/processed/player_matches.parquet')

#print(data.info())
# find min and max dates
cutoff = data['start_time'].quantile(0.85)
#print(pd.to_datetime(cutoff, unit = "s"))

test_data = data[data.start_time > cutoff]
training_data = data[data.start_time <= cutoff]

print(test_data.info())
print(training_data.info())

#).rename(columns = {'radiant_team_id' : 'team_id'})
test_teams = pd.concat([test_data['radiant_team_id'],test_data['dire_team_id']]).drop_duplicates()
training_teams = pd.concat([training_data['radiant_team_id'],training_data['dire_team_id']]).drop_duplicates()
x = test_teams[~test_teams.isin(training_teams)].drop_duplicates()
y = training_teams[~training_teams.isin(test_teams)].drop_duplicates()



z = test_data[test_data['radiant_team_id'].isin(x)].radiant_team_id.value_counts()
print(z.head(100).sum())
print(z.median())
print(len(test_data[test_data['radiant_team_id'].isin(x) | test_data['dire_team_id'].isin(x)]) / len(test_data))

