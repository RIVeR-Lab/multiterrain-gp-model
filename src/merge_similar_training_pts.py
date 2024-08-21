#!/usr/bin/env python3

import pandas as pd

# File paths for the original CSV files
asphalt_csv = '/home/ananya/Code/multiterrain-gp-model/data/Asphalt/GP_Datasets/matched_gp_cmds.csv'
grass_csv = '/home/ananya/Code/multiterrain-gp-model/data/Grass/GP_Datasets/matched_gp_cmds.csv'
tile_csv = '/home/ananya/Code/multiterrain-gp-model/data/Tile/GP_Datasets/matched_gp_cmds.csv'

# Load the data
asphalt_df = pd.read_csv(asphalt_csv)
grass_df = pd.read_csv(grass_csv)
tile_df = pd.read_csv(tile_csv)

# Rename the output columns for each terrain to avoid overlap
asphalt_df = asphalt_df.rename(columns={
    'op_lin_error': 'Asphalt_Lin_Error',
    'op_ang_error': 'Asphalt_Ang_Error'
})

grass_df = grass_df.rename(columns={
    'op_lin_error': 'Grass_Lin_Error',
    'op_ang_error': 'Grass_Ang_Error'
})

tile_df = tile_df.rename(columns={
    'op_lin_error': 'Tile_Lin_Error',
    'op_ang_error': 'Tile_Ang_Error'
})

# Merge the dataframes on the input columns
combined_df = pd.merge(asphalt_df, grass_df, on=['ip_cmd_lin', 'ip_cmd_ang', 'ip_curr_lin', 'ip_curr_ang'])
combined_df = pd.merge(combined_df, tile_df, on=['ip_cmd_lin', 'ip_cmd_ang', 'ip_curr_lin', 'ip_curr_ang'])

# Save the combined dataframe to a new CSV file
combined_df.to_csv('/home/ananya/Code/multiterrain-gp-model/data/combined_terrain_errors/combined_terrain_errors.csv', index=False)

print("Combined CSV file created successfully.")
