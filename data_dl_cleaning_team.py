# In this script we will clean the df before exploring datas

###
# Packages
###
import numpy as np
import pandas as pd
import data_dl_cleaning_team_func as dl_clean_func

###
# Packages Options
###
pd.options.mode.chained_assignment = None

###
# INPUT
###
season = '2020-2021'
team = 'GSW'

###
# DL & direct filter of datas (as we only want to study players' shots)
###
df_full_datas = dl_clean_func.import_plays_datas(season=season)
df_full_datas = df_full_datas[(df_full_datas['event_type'] == 'shot') &
                              (df_full_datas['team'] == team) &
                              (df_full_datas['period'] <= 4)]

###
# Merge of players' info to get with original datas
###
df_players = dl_clean_func.import_players_datas(season2=season)
df_full_datas = pd.merge(df_full_datas, df_players,
                         how='left',
                         on='player')

###
# New df with filtered columns: df_full_datas2
###
df_full_datas2 = df_full_datas[[
    'game_id', 'data_set', 'event_type',
    'a1', 'a2', 'a3', 'a4', 'a5',
    'h1', 'h2', 'h3', 'h4', 'h5',
    'team', 'away_score', 'home_score',
    'period', 'play_length', 'elapsed',
    'player', 'assist', 'result', 'points',
    'shot_distance', 'converted_x', 'converted_y', 'type',
    'Position', 'Experience', 'Age']]

###
# Cleaning play_length variable: Transform time strings into doubles
###
df_full_datas2['play_length'] = df_full_datas2.apply(
    lambda row: int(row['play_length'][-2:]),
    axis=1
)

# We notice play_length > 24 secs which is an anomaly, we eliminate them from df
df_full_datas2 = df_full_datas2[df_full_datas2['play_length'] <= 24]

# In addition, 1129 lines does not have coordinates, we eliminate them as well
df_full_datas2 = df_full_datas2[~((df_full_datas2['converted_x'].isna()) | (df_full_datas2['converted_y'].isna()))]

###
# Cleaning elapsed variable: Transform time strings into doubles
###
df_full_datas2['elapsed_quarter'] = df_full_datas2.apply(
    lambda row: dl_clean_func.elapsed_to_float_quarter(row['elapsed']),
    axis=1
)

df_full_datas2['elapsed_game'] = df_full_datas2.apply(
    lambda row: dl_clean_func.elapsed_to_float_total(row['period'], row['elapsed']),
    axis=1
)

###
# Cleaning result variable: Transform string result into integer
###
df_full_datas2['result'] = df_full_datas2['result'].apply(
    lambda row: 1 if row == 'made' else 0
)

###
# Cleaning assist variable: Transform string assist variable into integer
###
df_full_datas2['assist'] = df_full_datas2['assist'].apply(
    lambda row: 0 if pd.isnull(row) else 1
)

###
# Cleaning & Feature: Re-arrange x and y coordinates & define area regarding shots coordinates
###
df_full_datas2['x_shot'] = df_full_datas2.apply(
    lambda row: dl_clean_func.x_shots(row['converted_x'], row['converted_y'], row['shot_distance']),
    axis=1
)

df_full_datas2['y_shot'] = df_full_datas2.apply(
    lambda row: dl_clean_func.y_shots(row['converted_y'], row['shot_distance']),
    axis=1
)

df_full_datas2['area_shot'] = df_full_datas2.apply(
    lambda row: dl_clean_func.area_defined(row['shot_distance'], row['x_shot'], row['y_shot'], row['type']),
    axis=1
)

# Y : We classify areas (3pt have value 3, paint shots value 1 and the rest 2)
df_full_datas2['area_shot_number'] = df_full_datas2.apply(
    lambda row: dl_clean_func.area_defined_number(row['area_shot']),
    axis=1
)

# Y Lag values (lag 1 & 2)
df_full_datas2['area_shot_number_lag1'] = df_full_datas2[['game_id', 'area_shot_number']].groupby(
    by=['game_id'],
).shift(1)
df_full_datas2.loc[df_full_datas2['area_shot_number_lag1'].isna(), 'area_shot_number_lag1'] = int(0)

df_full_datas2['area_shot_number_lag2'] = df_full_datas2[['game_id', 'area_shot_number']].groupby(
    by=['game_id'],
).shift(2)
df_full_datas2.loc[df_full_datas2['area_shot_number_lag2'].isna(), 'area_shot_number_lag2'] = int(0)

###
# Feature: Check if a player plays at home or away & calculate the game scoring difference
# before a player make the bucket or not
###
df_full_datas2['home'] = df_full_datas2.apply(
    lambda row: 1 if row['player'] in '%s-%s-%s-%s-%s' % (row['h1'], row['h2'], row['h3'], row['h4'], row['h5']) else 0,
    axis=1
)

df_full_datas2['away'] = df_full_datas2.apply(
    lambda row: 1 if row['home'] == 0 else 0,
    axis=1
)

df_full_datas2['team_scorediff'] = (df_full_datas2['home'] - df_full_datas2['away']) * \
                                            (df_full_datas2['home_score'] - df_full_datas2['away_score']) - \
                                            df_full_datas2['points']

###
# Feature: Calculate players' total points during a season and during a game
###
df_full_datas2['team_game_points'] = df_full_datas2.groupby(
    by=['game_id']
)['points'].cumsum() - df_full_datas2['points']

###
# Feature: We intialize next calculations by creating a variable that will help us to count nb of shots taken by players
###
df_full_datas2['shots'] = np.ones(
    (len(df_full_datas2), 1),
    dtype=np.int8
)

df_full_datas2['3shots'] = df_full_datas2['area_shot_number'].apply(
    lambda row: 1 if row == 3 else 0
)

df_full_datas2['3shots_made'] = df_full_datas2.apply(
    lambda row: 1 if (row['3shots'] == 1 & row['result'] == 1) else 0,
    axis=1
)

###
# Feature: Calculate players' Field Goal percentage before shot is taken but first we change result variable into int
###
# 2pts
df_full_datas2['shots_team_made'] = df_full_datas2.groupby(
    by=['game_id']
)['result'].cumsum() - df_full_datas2['result']

df_full_datas2['shots_team_total'] = df_full_datas2.groupby(
    by=['game_id']
)['shots'].cumsum() - 1

df_full_datas2['FG'] = round(
    df_full_datas2['shots_team_made'] / df_full_datas2['shots_team_total'] * 100,
    2
)
df_full_datas2.loc[df_full_datas2['FG'].isna(), 'FG'] = float(0)

# 3pts
df_full_datas2['3shots_team_total'] = df_full_datas2.groupby(
    by=['game_id']
)['3shots'].cumsum() - df_full_datas2['3shots']

df_full_datas2['3shots_team_made'] = df_full_datas2.groupby(
    by=['game_id']
)['3shots_made'].cumsum() - df_full_datas2['3shots_made']

df_full_datas2['3FG'] = round(
    df_full_datas2['3shots_team_made'] / df_full_datas2['3shots_team_total'] * 100,
    2
)
df_full_datas2.loc[df_full_datas2['3FG'].isna(), '3FG'] = float(0)

###
# Feature: Calculate nb of shots consecutively made/missed by player before taking a shot
###
# We create a new df to re-ordered by player and time to evaluate a line in function of the preceding in a for loop
df_players_shots = df_full_datas2[['game_id', 'period', 'elapsed', 'result']]
df_players_shots = df_players_shots.sort_values(['game_id', 'period', 'elapsed']).reset_index()

# We initialize a new variable column by creating an array full of zero that we will complete in a for loop
df_players_shots['team_streak'] = np.zeros((len(df_players_shots), 1))
df_players_shots['first_game_bucket'] = np.zeros((len(df_players_shots), 1))
df_players_shots.at[0, 'first_game_bucket'] = 1

for i in range(1, len(df_players_shots)):

    if df_players_shots.at[i, 'game_id'] != df_players_shots.at[i - 1, 'game_id']:

        df_players_shots.at[i, 'team_streak'] = 0
        df_players_shots.at[i, 'first_game_bucket'] = 1
    else:
        if (df_players_shots.at[i - 1, 'team_streak'] >= 0) & \
                (df_players_shots.at[i - 1, 'result'] == 0):

            df_players_shots.at[i, 'team_streak'] = -1
        elif (df_players_shots.at[i - 1, 'team_streak'] <= 0) & \
                (df_players_shots.at[i - 1, 'result'] == 1):

            df_players_shots.at[i, 'team_streak'] = 1
        elif (df_players_shots.at[i - 1, 'team_streak'] > 0) & \
                (df_players_shots.at[i - 1, 'result'] == 1):

            df_players_shots.at[i, 'team_streak'] = df_players_shots.at[i - 1, 'team_streak'] + 1
        elif (df_players_shots.at[i - 1, 'team_streak'] < 0) & \
                (df_players_shots.at[i - 1, 'result'] == 0):

            df_players_shots.at[i, 'team_streak'] = df_players_shots.at[i - 1, 'team_streak'] - 1

# We merge this new df to the main one to aggregate player's shooting streak
df_full_datas2 = pd.merge(
    df_full_datas2, df_players_shots[['game_id', 'period', 'elapsed', 'team_streak', 'first_game_bucket']],
    how='left',
    on=['game_id', 'period', 'elapsed']
)

###
# Feature: Calculate number of assists/non assists received by a player before taking a shot
###
df_full_datas2['assist_team_total'] = df_full_datas2.groupby(
    by=['game_id']
)['assist'].cumsum() - df_full_datas2['assist']

df_full_datas2['ratio_assist_unassisted'] = round(
    df_full_datas2['assist_team_total'] / df_full_datas2['shots_team_made'] * 100,
    2
)

df_full_datas2.loc[df_full_datas2['ratio_assist_unassisted'].isna(), 'ratio_assist_unassisted'] = float(0)

###
# Last Cleaning: We modify elapsed columns by creating intervals instead of float numbers
###
# df_full_datas2['elapsed'] = df_full_datas2['elapsed'].apply(dl_clean_func.interval_defined)

###
# Cleaning: Categorization of players with multiple positions
###
df_full_datas2['Position'] = df_full_datas2['Position'].apply(dl_clean_func.position_defined)

###
# Last Cleaning: We create values to check players on the floor
###
df_roster = dl_clean_func.import_players_dico(season=season)
unique_players_vect = np.unique(df_roster.loc[df_roster['team'] == team, 'player'])

df_full_datas3 = pd.concat([df_full_datas2,
                           pd.DataFrame(
                               np.zeros((
                                   len(df_full_datas2),
                                   len(unique_players_vect)
                               )),
                               columns=unique_players_vect)],
                           axis=1)

for i in unique_players_vect:
    df_full_datas3[i] = df_full_datas3.apply(
        lambda row: 1 if i in '%s-%s-%s-%s-%s-%s-%s-%s-%s-%s' % (
        row['h1'], row['h2'], row['h3'], row['h4'], row['h5'], row['a1'], row['a2'], row['a3'], row['a4'], row['a5'])
        else 0,
        axis=1
    )

###
# Feature: We keep only the following variables
###
df_final_datas = pd.concat([df_full_datas3[['result', 'area_shot', 'area_shot_number',
                                            'area_shot_number_lag1', 'area_shot_number_lag2',
                                            'home', 'team_scorediff',
                                            'elapsed', 'elapsed_quarter', 'elapsed_game',
                                            'period',  'first_game_bucket',
                                            'play_length', 'shot_distance', 'team_game_points',
                                            'shots_team_made', 'shots_team_total', 'FG',
                                            '3shots_team_made', '3shots_team_total', '3FG',
                                            'team_streak', 'assist_team_total', 'ratio_assist_unassisted',
                                            'Position', 'Experience', 'Age']],
                            df_full_datas3[unique_players_vect]
                            ], axis=1
                           )

df_final_datas = df_full_datas3[['result', 'area_shot', 'area_shot_number',
                                 'area_shot_number_lag1', 'area_shot_number_lag2',
                                 'home', 'team_scorediff',
                                 'elapsed', 'elapsed_quarter', 'elapsed_game',
                                 'period',  'first_game_bucket',
                                 'play_length', 'shot_distance', 'team_game_points',
                                 'shots_team_made', 'shots_team_total', 'FG',
                                 '3shots_team_made', '3shots_team_total', '3FG',
                                 'team_streak', 'assist_team_total', 'ratio_assist_unassisted',
                                 'Position', 'Experience', 'Age']]

###
# Conclusion: We write the df into a csv file in the corresponding season directory
###
dl_clean_func.export_df_cleaned(df_final_datas, season)
