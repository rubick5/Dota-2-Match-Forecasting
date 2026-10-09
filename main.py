import pandas as pd

data = pd.read_parquet('data/processed/data.parquet')
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


print("Table of teams in the test set, not in the training set")
print(x.info())
print("x.nunique():")
print(x.nunique())
print("\n\n\n")
print("Table of teams in the training set, not in the test set")
print(y.info())

z = test_data[test_data['radiant_team_id'].isin(x)].radiant_team_id.value_counts()
print(z.head(100).sum())
print(z.median())
print(len(test_data[test_data['radiant_team_id'].isin(x) | test_data['dire_team_id'].isin(x)]) / len(test_data))