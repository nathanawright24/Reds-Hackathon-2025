# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 09:24:00 2025

@author: natha
"""

#-------------------------------------------------------------------------------
#--------------------- Cincinnati Reds Hackathon 2025 --------------------------
#-------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib as mpl
import seaborn as sea
import matplotlib.pyplot as plt

savantdata = pd.read_csv("C:/Users/natha/Documents/BGA/Competitions/RedsHack25/savant_data_2021_2023.csv")
lahmandata = pd.read_csv("C:/Users/natha/Documents/BGA/Competitions/RedsHack25/lahman_people.csv")
batters = savantdata['batter']
uniquebatters = batters.unique

import pandas as pd

savantdata["has_risp"] = savantdata["on_2b"].notna() | savantdata["on_3b"].notna()

power_quotient_stats = savantdata.groupby(["batter", "launch_speed_angle", "has_risp"]).agg(
    avg_launch_speed=("launch_speed", "mean"),
    avg_launch_angle=("launch_angle", "mean"),
    avg_est_ba=("estimated_ba_using_speedangle", "mean"),
    avg_est_woba=("estimated_woba_using_speedangle", "mean"),
    avg_iso_value=("iso_value", "mean"),
    avg_hit_distance=("hit_distance_sc", "mean"),
    launch_speed_angle_count=("launch_speed_angle", "count")     
).reset_index()

total_counts = power_quotient_stats.groupby(["batter", "has_risp"])["launch_speed_angle_count"].transform("sum")
power_quotient_stats["launch_speed_angle_frequency"] = power_quotient_stats["launch_speed_angle_count"] / total_counts
power_quotient_stats.drop(columns=["launch_speed_angle_count"], inplace=True)

weighted_avgs = power_quotient_stats.groupby(["batter", "has_risp"]).apply(
    lambda x: pd.Series({
        "w_avg_lnch_spd": (x["avg_launch_speed"] * x["launch_speed_angle_frequency"]).sum() / x["launch_speed_angle_frequency"].sum(),
        "w_avg_lnch_ang": (x["avg_launch_angle"] * x["launch_speed_angle_frequency"]).sum() / x["launch_speed_angle_frequency"].sum(),
        "w_avg_est_ba": (x["avg_est_ba"] * x["launch_speed_angle_frequency"]).sum() / x["launch_speed_angle_frequency"].sum(),
        "w_avg_est_woba": (x["avg_est_woba"] * x["launch_speed_angle_frequency"]).sum() / x["launch_speed_angle_frequency"].sum(),
        "w_avg_iso_value": (x["avg_iso_value"] * x["launch_speed_angle_frequency"]).sum() / x["launch_speed_angle_frequency"].sum(),
        "w_avg_hit_distance": (x["avg_hit_distance"] * x["launch_speed_angle_frequency"]).sum() / x["launch_speed_angle_frequency"].sum()
    })
).reset_index()

weighted_avgs = weighted_avgs.pivot(index="batter", columns="has_risp", values=[
    "w_avg_lnch_spd", "w_avg_lnch_ang", "w_avg_est_ba", "w_avg_est_woba", "w_avg_iso_value", "w_avg_hit_distance"
])

weighted_avgs.columns = [f"{col[0]}_{'w_risp' if col[1] else 'wo_risp'}" for col in weighted_avgs.columns]
weighted_avgs.reset_index(inplace=True)

savantdata = savantdata.merge(weighted_avgs, on="batter", how="left")
weighted_avgs=weighted_avgs.rename(columns={
    'batter':'player'}
    )
weighted_avgs.to_csv("C:/Users/natha/Documents/BGA/Competitions/RedsHack25/batterweightedavgs.csv")
print(weighted_avgs.info())
#-------------------------------------------------------------------------------
# percentiles of weighted averages
for col in weighted_avgs.columns[1:]:  
    weighted_avgs[f"{col}_pctl"] = weighted_avgs[col].rank(pct=True)

import pandas as pd

def calculate_percentile_table(df):
    percentile_cols = [col for col in df.columns if '_pctl' in col]
    value_cols = [col.replace('_pctl', '') for col in percentile_cols]

    percentiles = [10, 25, 50, 75, 90]
    percentile_table = {}

    for col in value_cols:
        percentile_table[col] = {p: df[col].quantile(p / 100) for p in percentiles}

    return pd.DataFrame(percentile_table).T  # Transpose for better readability

percentile_table = calculate_percentile_table(weighted_avgs)
print(percentile_table)
percentile_table.to_csv(r"C:\Users\natha\Documents\BGA\Competitions\RedsHack25\GeneratedData\percentiles.csv")
def format_for_google_docs(df):
    formatted_table = df.to_csv(sep='\t', index=True)  # Tab-separated values
    return formatted_table

table_text = format_for_google_docs(percentile_table)
print(table_text)
percentile_table.to_csv(r"C:\Users\natha\Documents\BGA\Competitions\RedsHack25\GeneratedData\percentile_table.txt", sep="\t", index=True)

from IPython.display import display, HTML

# Convert DataFrame to an HTML table
table_html = percentile_table.to_html()

# Display in Kaggle
display(HTML(table_html))
import pandas as pd

# Display table
pd.set_option("display.float_format", "{:.2f}".format)  # Optional: Format numbers
display(percentile_table)


import pandas as pd

def calculate_percentile_table(df):
    percentile_cols = [col for col in df.columns if '_pctl' in col]
    value_cols = [col.replace('_pctl', '') for col in percentile_cols]

    percentiles = [0, 25, 50, 75, 100]
    percentile_table = {}

    for col in value_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        # Define lower and upper bounds
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Filter out outliers
        filtered_data = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)][col]

        # Calculate percentiles on filtered data
        percentile_table[col] = {p: filtered_data.quantile(p / 100) for p in percentiles}

    return pd.DataFrame(percentile_table).T  # Transpose for better readability

percentile_table = calculate_percentile_table(weighted_avgs)
print(percentile_table)

    
"""
import matplotlib.pyplot as plt
import seaborn as sns

# Select percentile columns
percentile_cols = [col for col in weighted_avgs.columns if '_pctl' in col]
value_cols = [col.replace('_pctl', '') for col in percentile_cols]

# Create a long-format DataFrame
percentile_data = weighted_avgs.melt(id_vars=['player'], value_vars=percentile_cols, var_name='Stat', value_name='Percentile')
percentile_data['Percentile'] = percentile_data['Percentile'] * 100
value_data = weighted_avgs.melt(id_vars=['player'], value_vars=value_cols, var_name='Stat', value_name='Value')

# Align names
percentile_data['Stat'] = percentile_data['Stat'].str.replace('_pctl', '')
merged_data = percentile_data.merge(value_data, on=['player', 'Stat'])

# Plot
plt.figure(figsize=(12, 6))
sns.scatterplot(data=merged_data, x='Percentile', y='Value', hue='Stat', alpha=0.6)
plt.axvline(25, color='gray', linestyle='--', label='25th Percentile')
plt.axvline(50, color='black', linestyle='-', label='Median (50th)')
plt.axvline(75, color='gray', linestyle='--', label='75th Percentile')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel("Percentile")
plt.ylabel("Weighted Average Value")
plt.title("Percentiles vs Weighted Average Values for Different Stats")
plt.show()"""
#-------------------------------------------------------------------------------
print(savantdata["events"].unique())
"""savantdata["base_value"] = savantdata["events"].map({
    "home_run": 4,
    "triple": 3,
    "double": 2,
    "single": 1,
    "grounded_into_double_play": -1
}).fillna(0)

w_slugging = savantdata.groupby(["batter", "has_risp"]).apply(
    lambda x: pd.Series({
        "w_slugging": x["base_value"].sum() / (x["events"].isin(["single", "double", "triple", "home_run", "grounded_into_double_play"]).sum())
    })
).reset_index()

w_slugging = w_slugging.pivot(index="batter", columns="has_risp", values="w_slugging").reset_index()
w_slugging.columns = ["batter", "w_slugging_wo_risp", "w_slugging_w_risp"]
"""
#-------------------------------------------------------------------------------
# Merging external dataframes together
missed_game_stretches = pd.read_csv(r'C:\Users\natha\Documents\BGA\Competitions\RedsHack25\GeneratedData\missed_game_stretches.csv')
playing_time = pd.read_csv(r'C:\Users\natha\Documents\BGA\Competitions\RedsHack25\GeneratedData\playing_time.csv')
player_info = pd.read_csv(r'C:\Users\natha\Documents\BGA\Competitions\RedsHack25\GeneratedData\player_info.csv')
player_stats = pd.read_csv(r'C:\Users\natha\Documents\BGA\Competitions\RedsHack25\GeneratedData\player_stats.csv')
dropcolumn = ['Unnamed: 0']
player_info=player_info.rename(columns={
    'player_mlb_id':'player'}
    )

def drop_column_from_dfs(dropcolumn, *dfs):
    for df in dfs:
        if dropcolumn in df.columns:
            df.drop(columns=[dropcolumn], inplace=True)

drop_column_from_dfs('Unnamed: 0', missed_game_stretches, playing_time, player_info, player_stats)
# Merge the DataFrames
totalsDF = weighted_avgs.merge(missed_game_stretches, on='player', how='left') \
                        .merge(playing_time, on='player', how='left') \
                        .merge(player_info, on='player', how='left') \
                        .merge(player_stats, on='player', how='left')
totalsDF.to_csv("C:/Users/natha/Documents/BGA/Competitions/RedsHack25/GeneratedData/totalsDF.csv",index=False)

player_statcast_summary = pd.read_csv("C:/Users/natha/Documents/BGA/Competitions/RedsHack25/GeneratedData/player_statcast_summary.csv")
pitchers_spin_rate_with_changes = pd.read_csv("C:/Users/natha/Documents/BGA/Competitions/RedsHack25/GeneratedData/pitchers_spin_rate_with_changes.csv")
totalsDF = totalsDF.merge(player_statcast_summary, on='player', how='left') \
                   .merge(pitchers_spin_rate_with_changes, on='player', how='left')
                   
totalsDF = totalsDF.rename(columns={
    '2021':'2021spinrate',
    '2022':'2022spinrate',
    '2023':'2023spinrate',
    'change_2022':'spinchange2022',
    'change_2023':'spinchange2023'}
    )
#------------------------------------------------------------------------------------------
#--------------------------------Pitcher dummy variables-----------------------------------
#------------------------------------------------------------------------------------------
pitcherstuff = [
    'pitcher', 'pitch_type', 'release_pos_x', 'release_pos_z', 'release_speed', 
    'events', 'p_throws', 'type', 'pitch_number_appearance', 'pitcher_at_bat_number', 
    'sp_indicator', 'rp_indicator', 'times_faced', 'bat_score', 'post_bat_score', 
    'outs_when_up', 'estimated_ba_using_speedangle', 'estimated_woba_using_speedangle', 
    'pitch_number', 'at_bat_number', 'game_year', 'inning', 'has_risp', 'game_pk','batter'
]
pitchervariables = savantdata[pitcherstuff]

# Sort data to ensure batters are in proper order per pitcher per game
pitchervariables = pitchervariables.sort_values(by=['pitcher', 'game_year', 'game_pk', 'pitch_number_appearance','batter'])

# Define a function to count batters faced properly
def count_batters_faced(df):
    return (df['batter'] != df['batter'].shift()).sum()

# Aggregate counting stats
counting_stats = pitchervariables.groupby(['pitcher', 'game_year']).agg(
    innings_pitched=('events', lambda x: sum(is_out(e) for e in x) / 3),
    total_pitches_thrown=('pitch_number_appearance', 'max'),  # Max pitch number per game
    batters_faced=('batter', count_batters_faced),  # Count new batter appearances
    strikeouts=('events', lambda x: sum(is_strikeout(e) for e in x)),
    walks_allowed=('events', lambda x: sum(is_walk(e) for e in x)),
    hits_allowed=('events', lambda x: sum(is_hit(e) for e in x)),
    runs_allowed=('post_bat_score', 'max')  # Max post_bat_score to reflect total runs allowed
).reset_index()

# View results
print(counting_stats.head())

#-----------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#------------------------ Gradient Boost Modeling ------------------------------
#-------------------------------------------------------------------------------
batters = totalsDF[totalsDF['role']==1]
pitchers = totalsDF[totalsDF['role']==0]
batters.to_csv("C:/Users/natha/Documents/BGA/Competitions/RedsHack25/GeneratedData/battertotals.csv", index=False)
pitchers.to_csv("C:/Users/natha/Documents/BGA/Competitions/RedsHack25/GeneratedData/pitchertotals.csv", index=False)
print(batters.info())
!pip install xgboost lightgbm catboost

from itertools import chain, combinations
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LassoCV
import numpy as np

# Define independent (batterX) and dependent (batterY) variables
batterX_list = [
    'w_avg_lnch_spd_wo_risp', 'w_avg_lnch_spd_w_risp',
    'w_avg_lnch_ang_wo_risp', 'w_avg_lnch_ang_w_risp',
    'w_avg_est_ba_wo_risp', 'w_avg_est_ba_w_risp',
    'w_avg_est_woba_wo_risp', 'w_avg_est_woba_w_risp',
    'w_avg_iso_value_wo_risp', 'w_avg_iso_value_w_risp',
    'w_avg_hit_distance_wo_risp', 'w_avg_hit_distance_w_risp',
    'w_avg_lnch_spd_wo_risp_pctl', 'w_avg_lnch_spd_w_risp_pctl',
    'w_avg_lnch_ang_wo_risp_pctl', 'w_avg_lnch_ang_w_risp_pctl',
    'w_avg_est_ba_wo_risp_pctl', 'w_avg_est_ba_w_risp_pctl',
    'w_avg_est_woba_wo_risp_pctl', 'w_avg_est_woba_w_risp_pctl',
    'w_avg_iso_value_wo_risp_pctl', 'w_avg_iso_value_w_risp_pctl',
    'w_avg_hit_distance_wo_risp_pctl', 'w_avg_hit_distance_w_risp_pctl',
    'ten_days_off_2021_2022', 'ten_days_off_2022_2023',
    'games_played_2021_2022',
    'appearances_2021', 'appearances_2022',
    'avg_appearances_2021_2022', 'trend_appearances_2021_2022',
    'height', 'weight',
    'age_at_start_of_2023', 'years_in_mlb_by_2023',
    'obp_2021', 'obp_2022',
    'slg_2021', 'slg_2022',
    'ops_2021', 'ops_2022'
]

batterX = batters[batterX_list]


batterY = batters["appearances_2023"]  # Dependent variable
batterX = batterX.fillna(0)
# Step 1: Feature selection with LASSO
lasso = LassoCV(cv=5, random_state=42).fit(batterX, batterY)
selected_features = batterX.columns[lasso.coef_ != 0]
batterX = batterX[selected_features]

# Step 2: Generate all possible subsets of selected features
def get_feature_subsets(features):
    return chain.from_iterable(combinations(features, r) for r in range(1, len(features) + 1))

feature_subsets = list(get_feature_subsets(selected_features))

# Step 3: Initialize models
models = {
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(objective="reg:squarederror", random_state=42),
    "LightGBM": LGBMRegressor(random_state=42)
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

best_results = {}

# Step 4: Iterate over models and feature subsets
for model_name, model in models.items():
    best_rmse = float("inf")
    best_subset = None
    
    for subset in feature_subsets:
        X_subset = batterX[list(subset)]
        scores = cross_val_score(model, X_subset, batterY, cv=kf, scoring="neg_root_mean_squared_error")
        mean_rmse = -np.mean(scores)

        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_subset = subset
    
    best_results[model_name] = {"Best RMSE": best_rmse, "Best Features": best_subset}

# Print best feature subsets and RMSE for each model
for model_name, result in best_results.items():
    print(f"Model: {model_name}")
    print(f"Best RMSE: {result['Best RMSE']:.4f}")
    print(f"Best Features: {result['Best Features']}\n")
