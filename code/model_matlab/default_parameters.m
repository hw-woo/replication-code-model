% default_parameters.m
function [model_params, psi_d_range] = default_parameters()
    % Default range for psi_d
    psi_d_range = 4.5;  % Just use single value for testing

    % Set model parameters
    model_params = struct();
    model_params.n_points = 2;  % number of starting points for optimization
    model_params.seed = 0;      % random seed
    model_params.theta = 0;     % initial theta value
    
    % Add any other necessary parameters here
end