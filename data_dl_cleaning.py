# In this script we will clean the df before exploring datas

###
# Packages
###
import data_dl_cleaning_func as dl_clean_func
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

###
# INPUT
###
season = '2020-2021'

###
# DL & direct filter of datas (as we only want to study players' shots)
###
df_play = dl_clean_func.import_plays_datas(season=season)
df_play = df_play[df_play['event_type'] == 'shot']

###
# Merge of players' info to get with original datas
###
df_players = dl_clean_func.import_players_datas(season2=season)
df_players[['Age', 'Experience']] = df_players[['Age', 'Experience']].astype('int8')

df_full_datas = pd.merge(df_play, df_players,
                         how='left',
                         on='player')

# Once merged, we can delete both df_play and df_players for memory optimization
del df_play, df_players

###
# Check if a player is missing
###
if len(df_full_datas[df_full_datas['Age'].isna()]) == 0:
    print('No missing player in players infos files')

###
# Study of df_full_datas architectures
###
print('Le df possède {0} individus et {1} variables'.format(df_full_datas.shape[0], df_full_datas.shape[1]))

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
    'Position', 'Experience', 'Age', 'Salary']]

###
# Cleaning play_length variable: Transform time strings into doubles
###
df_full_datas2['play_length'] = df_full_datas2.apply(
    lambda row: int(row['play_length'][-2:]),
    axis=1
)
df_full_datas2['play_length'] = df_full_datas2['play_length'].astype('int8')

# We notice play_length > 24 secs which is an anomaly, we eliminate them from df
df_full_datas2 = df_full_datas2[df_full_datas2['play_length'] <= 24]

# In addition, 1129 lines does not have coordinates, we eliminate them as well
df_full_datas2 = df_full_datas2[~((df_full_datas2['converted_x'].isna()) | (df_full_datas2['converted_y'].isna()))]

# We translate certain variables to integers for speed calculation purposes
for i in ['away_score', 'home_score', 'period', 'points', 'shot_distance', 'converted_x', 'converted_y']:
    df_full_datas2[i] = df_full_datas2[i].astype('int8')

###
# Cleaning elapsed variable: Transform time strings into doubles
###
df_full_datas2['elapsed'] = df_full_datas2.apply(
    lambda row: dl_clean_func.elapsed_to_float(row['period'], row['elapsed']),
    axis=1
)
df_full_datas2['elapsed'] = df_full_datas2['elapsed'].astype('float16')

###
# Cleaning result variable: Transform string result into integer
###
df_full_datas2['result'] = df_full_datas2['result'].apply(
    lambda row: 1 if row == 'made' else 0
)
df_full_datas2['result'] = df_full_datas2['result'].astype('int8')

###
# Cleaning assist variable: Transform string assist variable into integer
###
df_full_datas2['assist'] = df_full_datas2['assist'].apply(
    lambda row: 0 if pd.isnull(row) else 1
)
df_full_datas2['assist'] = df_full_datas2['assist'].astype('int8')

###
# Cleaning & Feature: Re-arrange x and y coordinates & define area regarding shots coordinates
###
df_full_datas2['x_shot'] = df_full_datas2.apply(
    lambda row: dl_clean_func.x_shots(row['converted_x'], row['converted_y'], row['shot_distance']),
    axis=1
)
df_full_datas2['x_shot'] = df_full_datas2['x_shot'].astype('int8')

df_full_datas2['y_shot'] = df_full_datas2.apply(
    lambda row: dl_clean_func.y_shots(row['converted_y'], row['shot_distance']),
    axis=1
)
df_full_datas2['y_shot'] = df_full_datas2['y_shot'].astype('int8')

df_full_datas2['area_shot'] = df_full_datas2.apply(
    lambda row: dl_clean_func.area_defined(row['shot_distance'], row['x_shot'], row['y_shot'], row['type']),
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
df_full_datas2['player_home'] = df_full_datas2['player_home'].astype('int8')

df_full_datas2['player_away'] = df_full_datas2.apply(
    lambda row: 1 if row['player_home'] == 0 else 0,
    axis=1
)
df_full_datas2['player_away'] = df_full_datas2['player_away'].astype('int8')

df_full_datas2['player_team_scorediff'] = (df_full_datas2['player_home'] - df_full_datas2['player_away']) * \
                                            (df_full_datas2['home_score'] - df_full_datas2['away_score']) - \
                                            df_full_datas2['points']
df_full_datas2['player_team_scorediff'] = df_full_datas2['player_team_scorediff'].astype('int8')

###
# Feature: Calculate players' total points during a season and during a game
###
df_full_datas2['player_game_points'] = df_full_datas2.groupby(
    by=['game_id', 'player']
)['points'].cumsum() - df_full_datas2['points']
df_full_datas2['player_game_points'] = df_full_datas2['player_game_points'].astype('int16')

df_full_datas2['player_season_points'] = df_full_datas2.groupby(
    by=['player']
)['points'].cumsum() - df_full_datas2['points']
df_full_datas2['player_season_points'] = df_full_datas2['player_season_points'].astype('int16')

###
# Feature: We intialize next calculations by creating a variable that will help us to count nb of shots taken by players
###
df_full_datas2['shots'] = np.ones(
    (len(df_full_datas2), 1),
    dtype=np.int8
)

###
# Feature: Calculate players' Field Goal percentage before shot is taken but first we change result variable into int
###
df_full_datas2['shots_player_made'] = df_full_datas2.groupby(
    by=['game_id', 'player']
)['result'].cumsum() - df_full_datas2['result']
df_full_datas2['shots_player_made'] = df_full_datas2['shots_player_made'].astype('int8')

df_full_datas2['shots_player_total'] = df_full_datas2.groupby(
    by=['game_id', 'player']
)['shots'].cumsum() - 1
df_full_datas2['shots_player_total'] = df_full_datas2['shots_player_total'].astype('int8')

df_full_datas2['FG_player'] = round(
    df_full_datas2['shots_player_made'] / df_full_datas2['shots_player_total'] * 100,
    2
)
df_full_datas2['FG_player'] = df_full_datas2['FG_player'].astype('float16')

df_full_datas2.loc[df_full_datas2['FG_player'].isna(), 'FG_player'] = float(0)

###
# Feature: Calculate nb of shots consecutively made/missed by player before taking a shot
###
# We create a new df to re-ordered by player and time to evaluate a line in function of the preceding in a for loop
df_players_shots = df_full_datas2[['game_id', 'elapsed', 'player', 'result']]
df_players_shots = df_players_shots.sort_values(['player', 'game_id', 'elapsed']).reset_index()

# We initialize a new variable column by creating an array full of zero that we will complete in a for loop
df_players_shots['player_streak'] = np.empty((len(df_players_shots), 1))
df_players_shots['player_streak'] = df_players_shots['player_streak'].astype('int8')
for i in range(1, len(df_players_shots)):

    if (df_players_shots.at[i, 'player'] != df_players_shots.at[i - 1, 'player']) | \
            (df_players_shots.at[i, 'game_id'] != df_players_shots.at[i - 1, 'game_id']):

        df_players_shots.at[i, 'player_streak'] = 0

    else:
        if (df_players_shots.at[i - 1, 'player_streak'] >= 0) & \
                (df_players_shots.at[i - 1, 'result'] == 0):

            df_players_shots.at[i, 'player_streak'] = -1

        elif (df_players_shots.at[i - 1, 'player_streak'] <= 0) & \
                (df_players_shots.at[i - 1, 'result'] == 1):

            df_players_shots.at[i, 'player_streak'] = 1

        elif (df_players_shots.at[i - 1, 'player_streak'] > 0) & \
                (df_players_shots.at[i - 1, 'result'] == 1):

            df_players_shots.at[i, 'player_streak'] = df_players_shots.at[i - 1, 'player_streak'] + 1

        elif (df_players_shots.at[i - 1, 'player_streak'] < 0) & \
                (df_players_shots.at[i - 1, 'result'] == 0):

            df_players_shots.at[i, 'player_streak'] = df_players_shots.at[i - 1, 'player_streak'] - 1

# We merge this new df to the main one to aggregate player's shooting streak
df_full_datas2 = pd.merge(
    df_full_datas2, df_players_shots[['game_id', 'player', 'elapsed', 'player_streak']],
    how='left',
    on=['game_id', 'player', 'elapsed']
)

###
# Feature: Calculate number of assists/non assists received by a player before taking a shot
###
df_full_datas2['assist_player_total'] = df_full_datas2.groupby(
    by=['game_id', 'player']
)['assist'].cumsum() - df_full_datas2['assist']
df_full_datas2['assist_player_total'] = df_full_datas2['assist_player_total'].astype('int8')

df_full_datas2['ratio_assist_player'] = round(
    df_full_datas2['assist_player_total'] / df_full_datas2['shots_player_made'] * 100,
    2
)
df_full_datas2['ratio_assist_player'] = df_full_datas2['ratio_assist_player'].astype('float16')

df_full_datas2.loc[df_full_datas2['ratio_assist_player'].isna(), 'ratio_assist_player'] = float(0)

###
# Last Cleaning: We modify elapsed columns by creating intervals instead of float numbers
###
df_full_datas2['elapsed'] = df_full_datas2['elapsed'].apply(dl_clean_func.interval_defined)

###
# Last Cleaning: Categorization of players with multiple positions
###
df_full_datas2['Position'] = df_full_datas2['Position'].apply(dl_clean_func.position_defined)

###
# Feature: We keep only the following variables
###
df_final_datas = df_full_datas2[['result',
                                 'away_score', 'home_score', 'player_team_scorediff',
                                 'play_length', 'elapsed',
                                 'shot_distance', 'area_shot',
                                 'player_home', 'player_away', 'player_season_points', 'player_game_points',
                                 'shots_player_made', 'shots_player_total', 'FG_player', 'player_streak',
                                 'assist_player_total', 'ratio_assist_player',
                                 'Position', 'Experience', 'Age']]

###
# Conclusion: We write the df into a csv file in the corresponding season directory
###
dl_clean_func.export_df_cleaned(df_final_datas, season)
