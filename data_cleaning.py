# In this script we will clean the df before exploring datas

###
# Packages
###
import data_dl_func as dl_func
import data_cleaning_func as clean_func
import pandas as pd

###
# DL & direct filter of datas (as we only want to study players' shots)
###
season = '2020-2021'
df_play = dl_func.import_plays_datas(season=season)
df_play = df_play[df_play['event_type'] == 'shot']

###
# Merge of players' info to get with original datas
###
df_players = dl_func.import_players_datas(season2=season)
df_full_datas = pd.merge(df_play, df_players,
                         how='left',
                         on='player')

###
# Check if a player is missing
###
if len(df_full_datas[df_full_datas['Age'].isna()]) == 0:
    print('No missing player in players infos files')

###
# Study of df_full_datas architectures
###
print('Le df possède {0} individus et {1} variables'.format(df_full_datas.shape[0], df_full_datas.shape[1]))
df_full_datas.columns

###
# New df with filtered columns: df_full_datas2
###
df_full_datas2 = df_full_datas[[
    'game_id', 'data_set', 'event_type',
    'h1', 'h2', 'h3', 'h4', 'h5',
    'team', 'away_score', 'home_score',
    'period', 'play_length', 'elapsed',
    'player', 'assist', 'result', 'points',
    'shot_distance', 'converted_x', 'converted_y', 'type',
    'Position', 'Experience', 'Age', 'Salary'
]]

###
# Cleaning play_length variable: Transform time strings into doubles
###
df_full_datas2['play_length'] = df_full_datas2['play_length'].apply(
    lambda row: int(row[-2:])
)
df_full_datas2['play_length'].value_counts().sort_index().plot(kind='bar')

# We notice play_length > 24 secs which is an anomaly, we eliminate them from df
df_full_datas2 = df_full_datas2[df_full_datas2['play_length'] <= 24]

###
# Cleaning elapsed variable: Transform time strings into doubles
###
df_full_datas2['elapsed_float'] = df_full_datas2.apply(
    lambda row: clean_func.elapsed_to_float(row['period'], row['elapsed']),
    axis=1
)

###
# Cleaning & Feature: Re-arrange x and y coordinates & define area regarding shots coordinates
###
df_full_datas2['x_shot'] = df_full_datas2.apply(
    lambda row: clean_func.x_shots(row['converted_x'], row['converted_y'], row['shot_distance']),
    axis=1
)

df_full_datas2['y_shot'] = df_full_datas2.apply(
    lambda row: clean_func.y_shots(row['converted_y'], row['shot_distance']),
    axis=1
)

df_full_datas2['area_shot'] = df_full_datas2.apply(
    lambda row: clean_func.area_defined(row['shot_distance'], row['x_shot'], row['y_shot'], row['type']),
    axis=1
)

###
# Feature: Check if a player plays at home or away & calculate the game scoring difference
# before a player make the bucket or not
###
df_full_datas2['player_home'] = df_full_datas2.apply(
    lambda row: 1 if row['player'] in '%s-%s-%s-%s-%s' % (row['h1'], row['h2'], row['h3'], row['h4'], row['h5']) else 0,
    axis=1
)

df_full_datas2['player_away'] = df_full_datas2.apply(
    lambda row: 1
    if row['player'] not in '%s-%s-%s-%s-%s' % (row['h1'], row['h2'], row['h3'], row['h4'], row['h5'])
    else 0,
    axis=1
)

df_full_datas2['player_game_score'] = (df_full_datas2['player_home'] - df_full_datas2['player_away']) * \
                                      (df_full_datas2['home_score'] - df_full_datas2['away_score']) - \
                                      df_full_datas2['points']

###
# Feature: Calculate nb of shots consecutively made/missed by player before taking a shot
###


###
# Feature: Calculate nb of shots consecutively made/missed by player before taking a shot
###

###
# Feature: Calculate nb of shots previously made/missed by player during a game
###

###
# Feature: Calculate FG% per area by player during a game before taking a shot
###

###
# Feature: Calculate number of assists/non assists received by a player before taking a shot
###
