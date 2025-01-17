function estimate_model(savedir_root, psi_d, varargin)
    % Estimate model parameters
    %
    % Inputs:
    %   savedir_root - the root directory for saving results
    %   psi_d - psi_d parameter value
    %   varargin - optional parameters:
    %     theta (default: 0)
    %     options (struct with optional fields)
    
    % Convert inputs to appropriate types
    savedir_root = char(savedir_root);  % Ensure string type
    psi_d = double(psi_d);  % Ensure numeric type
    
    % Parse optional inputs
    if nargin > 2
        theta = double(varargin{1});
    else
        theta = 0;
    end
    
    if nargin > 3
        options = varargin{2};
    else
        options = struct('seed', 0, 'n_points', 6);
    end
    
    % Set flags
    estimate_theta = (theta == 0);
    
    % Create save directory path (using sprintf for path creation)
    if estimate_theta
        fprintf('\nESTIMATION (Estimate theta): psi_d = %.1f\n\n', psi_d);
        savedir = [savedir_root, filesep, sprintf('baseline_psi_d_%.1f', psi_d)];
    else
        fprintf('\nESTIMATION (Fix theta): psi_d = %.1f, theta = %.1f\n\n', psi_d, theta);
        savedir = [savedir_root, filesep, sprintf('fixtheta_psi_d_%.1f_theta_%.1f', psi_d, theta)];
    end
    
    % Create directories
    if ~exist(savedir, 'dir')
        mkdir(savedir);
    end
    if ~exist([savedir, filesep, 'temp'], 'dir')
        mkdir([savedir, filesep, 'temp']);
    end
    
    % Run estimation procedures
    try
        % Process input data
        [data_mom, data_occ, task_types, years] = process_data();
        
        % Setup model parameters
        model_params = setup_model_parameters(data_mom, data_occ, task_types, years);
        
        % Run main estimation
        estimate_parameters(model_params, savedir, psi_d, theta, options);
        
        fprintf('\nEstimation completed successfully\n');
        
    catch ME
        fprintf('Error in estimation: %s\n', ME.message);
        rethrow(ME);
    end
end

function [data_mom, data_occ, task_types, years] = process_data()
    % Load and process input data
    data_mom = csvread('../output/acs/moments_all.csv', 1, 0);  % Skip header row
    data_occ = csvread('../output/acs/moments_ztasks_broad.csv', 1, 0);  % Skip header row
    
    task_types = {'cont', 'abs', 'man', 'rout'};
    years = [1960, 1970, 1980, 1990, 2000, 2012, 2018];
end

function model_params = setup_model_parameters(data_mom, data_occ, task_types, years)
    % Setup model parameters
    model_params = struct();
    model_params.data_mom = data_mom;
    model_params.data_occ = data_occ;
    model_params.task_types = task_types;
    model_params.years = years;
end

function estimate_parameters(model_params, savedir, psi_d, theta, options)
    % Main estimation procedure
    % This is a placeholder - actual estimation code should go here
    fprintf('Running estimation with psi_d = %.2f\n', psi_d);
    if theta == 0
        fprintf('Estimating theta\n');
    else
        fprintf('Using fixed theta = %.2f\n', theta);
    end
end