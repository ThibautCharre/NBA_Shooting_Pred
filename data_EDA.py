# This scripts enables us to explore cleaned datas

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np

import os

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

###
# DL of cleaned datas
###
season = '2020-2021'
data_dir = '%s%s%s%s' % (os.getcwd(), '\\Datas\\', season, '\\df_shots_cleaned.csv')
df_cleaned = pd.read_csv(data_dir)

df_cleaned['new_result'] = df_cleaned['result'].apply(
    lambda row: 'made' if row == 1 else 'missed'
)

df_cleaned['elapsed'] = np.ceil(df_cleaned['elapsed']).astype('int16')

###
# 1st : General view of how baskets are made
###
df_shots_pos = pd.pivot_table(
    df_cleaned,
    index=['Position'],
    columns=['new_result'],
    aggfunc='size'
)
df_shots_pos['FG%'] = round(
    100 * df_shots_pos['made'] / (df_shots_pos['missed'] + df_shots_pos['made']),
    2
)

df_shots_shotdistance = pd.pivot_table(
    df_cleaned,
    index=['shot_distance'],
    columns=['new_result'],
    aggfunc='size', fill_value=0
)
df_shots_shotdistance['FG%'] = round(
    100 * df_shots_shotdistance['made'] / (df_shots_shotdistance['missed'] + df_shots_shotdistance['made']),
    2
)

df_shots_area = pd.pivot_table(
    df_cleaned,
    index=['area_shot'],
    columns=['new_result'],
    aggfunc='size'
)
df_shots_area['FG%'] = round(
    100 * df_shots_area['made'] / (df_shots_area['missed'] + df_shots_area['made']),
    2
)

df_shots_experience = pd.pivot_table(
    df_cleaned,
    index=['Experience'],
    columns=['new_result'],
    aggfunc='size'
)
df_shots_experience['FG%'] = round(
    100 * df_shots_experience['made'] / (df_shots_experience['missed'] + df_shots_experience['made']),
    2
)

df_shots_playtime = pd.pivot_table(
    df_cleaned,
    index=['play_length'],
    columns=['new_result'],
    aggfunc='size'
)
df_shots_playtime['FG%'] = round(
    100 * df_shots_playtime['made'] / (df_shots_playtime['missed'] + df_shots_playtime['made']),
    2
)

df_shots_gametime = pd.pivot_table(
    df_cleaned,
    index=['elapsed'],
    columns=['new_result'],
    aggfunc='size'
)
df_shots_gametime['FG%'] = round(
    100 * df_shots_gametime['made'] / (df_shots_gametime['missed'] + df_shots_gametime['made']),
    2
)

###
# 2nd : Heat Map of shot_distance with game
###
test = pd.pivot_table(df_cleaned[(df_cleaned['new_result'] == 'made') &
                                 (df_cleaned['area_shot'].str.contains('3pt')) &
                                 (df_cleaned['elapsed'] < 49)],
               index=['area_shot'],
               columns=['elapsed'],
               aggfunc='size', fill_value=0
               )
sns.heatmap(test, cmap='crest')

sns.boxplot(data=df_cleaned, x="player_streak", y="Position")

