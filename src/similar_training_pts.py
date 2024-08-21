#!/usr/bin/env python3

import pandas as pd
import numpy as np

# File paths for the original CSV files
asphalt_csv = '/home/ananya/Code/multiterrain-gp-model/data/Asphalt/GP_Datasets/unique_gp_cmds.csv'
grass_csv = '/home/ananya/Code/multiterrain-gp-model/data/Grass/GP_Datasets/unique_gp_cmds.csv'
tile_csv = '/home/ananya/Code/multiterrain-gp-model/data/Tile/GP_Datasets/unique_gp_cmds.csv'

# Load the data
asphalt_df = pd.read_csv(asphalt_csv)
grass_df = pd.read_csv(grass_csv)
tile_df = pd.read_csv(tile_csv)

# Round the input columns to one decimal place for comparison
input_columns = ['ip_cmd_lin', 'ip_cmd_ang', 'ip_curr_lin', 'ip_curr_ang']

asphalt_df_rounded = asphalt_df.copy()
asphalt_df_rounded[input_columns] = asphalt_df_rounded[input_columns].round(1)

grass_df_rounded = grass_df.copy()
grass_df_rounded[input_columns] = grass_df_rounded[input_columns].round(1)

tile_df_rounded = tile_df.copy()
tile_df_rounded[input_columns] = tile_df_rounded[input_columns].round(1)

# Find common rows where the rounded inputs match across all three dataframes
common_rows = pd.merge(asphalt_df_rounded, grass_df_rounded, on=input_columns, how='inner')
common_rows = pd.merge(common_rows, tile_df_rounded, on=input_columns, how='inner')

# Merge with original dataframes to get the corresponding output columns
asphalt_common = pd.merge(common_rows[input_columns], asphalt_df, on=input_columns, how='inner')
grass_common = pd.merge(common_rows[input_columns], grass_df, on=input_columns, how='inner')
tile_common = pd.merge(common_rows[input_columns], tile_df, on=input_columns, how='inner')

# Save the results to new CSV files
asphalt_common.to_csv('/home/ananya/Code/multiterrain-gp-model/data/Asphalt/GP_Datasets/matched_gp_cmds.csv', index=False)
grass_common.to_csv('/home/ananya/Code/multiterrain-gp-model/data/Grass/GP_Datasets/matched_gp_cmds.csv', index=False)
tile_common.to_csv('/home/ananya/Code/multiterrain-gp-model/data/Tile/GP_Datasets/matched_gp_cmds.csv', index=False)

print("Matched rows have been saved to the new CSV files.")
