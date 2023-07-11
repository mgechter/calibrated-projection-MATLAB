clear

rho_l = 0.95;
B = 600;


alpha = 0.1;


d = readtable("Data/full_dataset.csv");
d = d(:, ["mumbai" "bal" "max_premath_std" "max_postmath_std" "classyearid"]);
% omit mumbai treated group
d = d(d.mumbai == 0 | (d.mumbai == 1 & d.bal == 0), :);

tab_max_postmath_std = tabulate(d.max_postmath_std);
y_supp = tab_max_postmath_std(:,1);
n_supp = length(y_supp);

tab_max_premath_std = tabulate(d.max_premath_std);
x_supp = tab_max_premath_std(:,1);
n_x_supp = length(x_supp);


n = height(d);

c_e = d(d.mumbai == 0 & d.bal == 0, :).max_postmath_std;
t_e = d(d.mumbai == 0 & d.bal == 1, :).max_postmath_std;
c_a = d(d.mumbai == 1 & d.bal == 0, :).max_postmath_std;

p_a = sum(d.mumbai == 1)/n;
p_e = sum(d.mumbai == 0 & d.bal == 1)/sum(d.mumbai == 0);


load("Data/params_by_rho_l.mat")

rho_ls = keys(params_by_rho_l);
thetas = values(params_by_rho_l);

% get PPD theta
ppd_rho_l = 1;

theta_ppd = params_by_rho_l(ppd_rho_l);
theta_ppd = theta_ppd(1,:);

% test that moments compute
compute_moments_stdev(theta_ppd', y_supp, n_supp, d, p_a, p_e, ppd_rho_l, 1, n_x_supp)


% get the desired rho_l parameters
theta_lb_ub = params_by_rho_l(rho_l)

theta_ub = theta_lb_ub(2,:)';
theta_0 = theta_ub


[m_eq, m_ineq, m_eq_std, m_ineq_std] = compute_moments_stdev(theta_ub, y_supp, n_supp, d, p_a, p_e, rho_l, 1, n_x_supp);

theta_feas = [theta_ppd;
                theta_lb_ub];

            
LB_theta = [-3; zeros(n_x_supp - 1, 1); ...
            repmat([zeros(n_supp^2 - 1, 1); zeros(n_supp - 1, 1); zeros(n_supp^2 - 1, 1); zeros(n_supp^2, 1)], n_x_supp, 1)];
        
UB_theta = [3; ones(n_x_supp - 1, 1); ...
            repmat([ones(n_supp^2 - 1, 1); ones(n_supp - 1, 1); ones(n_supp^2 - 1, 1); ones(n_supp^2, 1)], n_x_supp, 1)];


conditional_param_restrictions = [ones(1, n_supp^2 - 1), zeros(1, n_supp - 1), zeros(1, n_supp^2 - 1), zeros(1, n_supp^2) ;
                                    zeros(1, n_supp^2 - 1), ones(1, n_supp - 1), zeros(1, n_supp^2 - 1), zeros(1, n_supp^2) ;
                                    zeros(1, n_supp^2 - 1), zeros(1, n_supp - 1), ones(1, n_supp^2 - 1), zeros(1, n_supp^2) ];
combined_conditional_param_restrictions = kron(eye(n_x_supp), conditional_param_restrictions);

% Require the PMF parameters to sum to 1 or less
            % xi
A_theta = [ 0, ones(1, n_x_supp - 1), ...
                    repmat([zeros(1, n_supp^2 - 1), zeros(1, n_supp - 1), zeros(1, n_supp^2 - 1), zeros(1, n_supp^2)], 1, n_x_supp) ;
            % conditional param restrictions
            zeros(height(combined_conditional_param_restrictions), 1), zeros(height(combined_conditional_param_restrictions), n_x_supp - 1), ...
                     combined_conditional_param_restrictions ];

        
b_theta = ones(height(A_theta), 1);

                                                                                                                      
p = [1; zeros(length(theta_0) - 1, 1)];


% KMSoptions

KMSoptions  = KMSoptions_Simulation();

KMSoptions.B            = B;
KMSoptions.seed         = 1;    % Seed value
KMSoptions.CVXGEN       = 0;    % Set equal to 1 if CVXGEN is used.  Set equal to 0 if CVX is used
KMSoptions.HR           = 1;    % use hit-and-run sampling
KMSoptions.numgrad      = true;             % Set equal to true to compute Dg using numerical gradients. 
KMSoptions.numgrad_steplength = eps^(1/3);  % step lenght of numericalg radient
KMSoptions.DGP          = 0;
KMSoptions.EAM_maxit = 50;
KMSoptions.parallel = 1;

[KMS_confidence_interval,KMS_output] = KMS_0_Main(d, theta_0, y_supp, n_supp, n_x_supp, p_a, p_e, rho_l, p, theta_feas, LB_theta, UB_theta, A_theta, b_theta, alpha, 'two-sided', 'AS' , NaN, NaN, [], KMSoptions);
                                                
% diagnostics
KMS_output

save(rho_l + "_" + alpha + ".mat");