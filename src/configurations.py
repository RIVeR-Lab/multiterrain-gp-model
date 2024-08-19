#! /usr/bin/python3

class ModelingParameters:
    """Configurations for all the scripts"""

    # Cutoff frequence in Hz for low pass position filter
    lpf_cutoff_hz = 1.5

    # Frequency in hz for the resampled and time synced cmd and loc data
    resampled_freq_hz = 10. ; resampled_dt_s = round(1./resampled_freq_hz, 3)

    # Distance between robot center of mass and center of the rear axle
    a_m = 0.145

    # Robot width in meters
    b_m = 0.377

    # Wheel radius in meters
    r_m = 0.09

    # Column names for commanded velocities
    cmd_names = ['V_linear_x (m/s)', 'V_linear_y (m/s)', 'V_linear_z (m/s)', 'V_ang_x (rad/s)', 'V_ang_y (rad/s)','V_ang_z (rad/s)', 'time (seconds)']

    # Column names for localization positions
    loc_names = ['time (seconds)', 'position_x (meters)', 'position_y (meters)', 'heading (rad)']

    # Low pass filter cutoff frequency when computing theta1 to 6
    theta_lpf_cutoff_hz = 1.0

    # Dataset types
    dataset_types = ["ConstantVel","Train","LongDistance"]

    # Types of terrains
    terrain_types = ["Asphalt", "Grass", "Tile"]

    # Learning rate for training GP Models
    lr = 0.01

    # Maximum number of GP training iterations
    max_opt_iter = 2000

    # Look back horizon when computing the weights for each terrain
    # Given the resampled_dt_s is around 100 ms, this corresponds
    # approximately to a look back time of 1.5 seconds
    look_back_horizon = 15

    # Types of GPs trained,linear and angular velocity error
    trained_gp_types_list = ["Linear","Angular"]

    # Number of GMM clusters to consider for training point selection
    gp_clustering_num_train_pts = 500

    # Number of GMM clusters to consider for testing point selection
    gp_clustering_num_test_pts = 500

    # Number of robot states
    num_robot_states = 5

    # Number of robot control actions
    num_robot_control_actions = 2

    # Selected uncertainty propagation method
    uncertainty_propagation_method = "NonLinear" #"Linear"

    # Contribution of the mean sigma point weight to the next state prediction
    # The contribution of each of the remaining 4*n, n=5 sigma points is thus:
    # (1.0 - mean_sigma_weight)/(4*5).
    mean_sigma_point_weight = 0.2

    # Penalizing the deviation between solutions iterations
    regularization_eps = 0.1