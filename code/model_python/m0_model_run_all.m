% m0_model_run_all.m (Octave version)
function m0_model_run_all
    % Set paths
    addpath(genpath(pwd));

    % Load default model parameters
    [model_params, psi_d_range] = default_parameters();

    % Run without parallel processing for Octave
    run_estimation(model_params, psi_d_range);
end

function run_estimation(model_params, psi_d_range)
    % Run the estimation
    for i = 1:length(psi_d_range)
        psi_d = psi_d_range(i);
        fprintf('Running estimation for psi_d = %.2f\n', psi_d);
        estimate_model(model_params, psi_d);
    end
end