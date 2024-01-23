# GOAL: Generate a Match prediction dataset from the game dataset, the games_detail dataset and the ranking dataset

# REMAINDER: Must Refactor the code!!!

import pandas as pd

games = pd.read_csv('./datasets/games.csv')
games_details = pd.read_csv('./datasets/games_details.csv')
ranking = pd.read_csv('./datasets/ranking.csv')

# Only consider the useful columns
games_details = games_details[['GAME_ID', 'TEAM_ID', 'PLAYER_ID', 'START_POSITION']]
# Since we have a large number of entries, we can drop all the rows with null values
games_details.dropna(inplace=True)

# Only consider the useful columns
# games = games[['HOME_TEAM_WINS', 'GAME_ID', 'GAME_DATE_EST', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'SEASON']]
# games = games.drop(
#     columns=['GAME_STATUS_TEXT', 'TEAM_ID_home', 'TEAM_ID_away', 'PTS_home', 'FG_PCT_home', 'FT_PCT_home',
#              'FG3_PCT_home', 'AST_home', 'REB_home'])
games = games.drop(columns=['GAME_STATUS_TEXT', 'TEAM_ID_home', 'TEAM_ID_away'])
# Join between the games and ranking, so we can extract the winrate of the home team before a match
merged = pd.merge(games, ranking, left_on=['HOME_TEAM_ID', 'GAME_DATE_EST'], right_on=['TEAM_ID', 'STANDINGSDATE'])
games['HOME_TEAM_Games'] = merged['G']
games['HOME_TEAM_Win_Rate'] = merged['W_PCT']

# Join between the games and ranking, so we can extract the winrate of the visitor team before a match
merged = pd.merge(games, ranking, left_on=['VISITOR_TEAM_ID', 'GAME_DATE_EST'], right_on=['TEAM_ID', 'STANDINGSDATE'])
games['VISITOR_TEAM_Games'] = merged['G']
games['VISITOR_TEAM_Win_Rate'] = merged['W_PCT']

# Join between the games and games_details, so we can extract the 5 main players of the home team
merged = pd.merge(games, games_details, left_on=['GAME_ID', 'HOME_TEAM_ID'], right_on=['GAME_ID', 'TEAM_ID'])

players = []

for game_id in merged['GAME_ID'].unique().tolist():
    rows = merged[merged['GAME_ID'] == game_id]
    players_id = rows['PLAYER_ID'].tolist()
    if len(players_id) == 5:
        players.append(rows['PLAYER_ID'].tolist())

df = pd.DataFrame(data=players, columns=['HOME_TEAM_player1', 'HOME_TEAM_player2', 'HOME_TEAM_player3',
                                         'HOME_TEAM_player4', 'HOME_TEAM_player5'], dtype='int64')

# Concatenate the home team's players with the games
dataframe = pd.concat([games, df], axis=1)

# Join between the games and games_details, so we can extract the 5 main players of the home team
merged = pd.merge(dataframe, games_details, left_on=['GAME_ID', 'VISITOR_TEAM_ID'],
                  right_on=['GAME_ID', 'TEAM_ID'])

players = []

for game_id in merged['GAME_ID'].unique().tolist():
    rows = merged[merged['GAME_ID'] == game_id]
    players_id = rows['PLAYER_ID'].tolist()
    if len(players_id) == 5:
        players.append(rows['PLAYER_ID'].tolist())

df = pd.DataFrame(data=players, columns=['VISITOR_TEAM_player1', 'VISITOR_TEAM_player2', 'VISITOR_TEAM_player3',
                                         'VISITOR_TEAM_player4', 'VISITOR_TEAM_player5'], dtype='int64')
# Concatenate the visitor team's players with the games
dataframe = pd.concat([dataframe, df], axis=1)

# Remove all entries which don't have a team formation
dataframe.dropna(inplace=True)

dataframe[['HOME_TEAM_player1', 'HOME_TEAM_player2', 'HOME_TEAM_player3', 'HOME_TEAM_player4', 'HOME_TEAM_player5',
           'VISITOR_TEAM_player1', 'VISITOR_TEAM_player2', 'VISITOR_TEAM_player3', 'VISITOR_TEAM_player4',
           'VISITOR_TEAM_player5']] = dataframe[
    ['HOME_TEAM_player1', 'HOME_TEAM_player2', 'HOME_TEAM_player3', 'HOME_TEAM_player4', 'HOME_TEAM_player5',
     'VISITOR_TEAM_player1', 'VISITOR_TEAM_player2', 'VISITOR_TEAM_player3', 'VISITOR_TEAM_player4',
     'VISITOR_TEAM_player5']].astype('int')

# Remove irrelevant data
dataframe.drop(columns=['GAME_DATE_EST'], inplace=True)

# Save the generated dataframe as csv file
dataframe.to_csv('./datasets/nba.csv', index=False)

dataframe.info()
