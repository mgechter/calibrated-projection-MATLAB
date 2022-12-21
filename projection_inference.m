clear

rho_l = 0.9;
B = 100;
alpha = 0.1;


d = readtable("Data/full_dataset.csv");
d = d(:, ["mumbai" "bal" "max_premath_std" "max_postmath_std" "classyearid"]);

% omit mumbai treated group
d = d(d.mumbai == 0 | (d.mumbai == 1 & d.bal == 0), :);


tab_max_premath_std = tabulate(d.max_premath_std);

conditions = tab_max_premath_std(:,1);
conditions = 0;

max_premath_std = conditions(1);

d_cond = d(d.max_premath_std == max_premath_std,:);

d = d_cond;

n = height(d);
kappa = sqrt(log(n));

c_e = d(d.mumbai == 0 & d.bal == 0, :).max_postmath_std;
t_e = d(d.mumbai == 0 & d.bal == 1, :).max_postmath_std;
c_a = d(d.mumbai == 1 & d.bal == 0, :).max_postmath_std;

tab_max_postmath_std = tabulate(c_e);

p_a = length(c_a)/length([c_e; t_e; c_a]);
p_e = length(t_e)/length([c_e; t_e]);
y_supp = tab_max_postmath_std(:,1);
n_supp = length(y_supp);



% now compute the moments themselves

% feasible values of parameters
beta_lb = 0.0427;
beta_ppd = 0.1374;

upsilon = tab_max_postmath_std(1:(n_supp -1),3) * 0.01;

pi = [0.5636,    0.0072,    0.0000,    0.0000,    0.0331,    0.2782,    0.0000,    0.0000,    0.0450,    0.0000,    0.0003,    0.0261, ...
    0.0000,    0.0000,    0.0464]';

pi_lb = [0.5617,    0.0000,    0.0091,    0.0000,    0.0800,    0.1676,    0.0376,    0.0261,    0.0000,    0.0715,    0.0000,    0.0000, ...
    0.0000, 0.0464, 0.0000]';
pi_lb_to_reshape = [pi_lb; 1 - sum(pi_lb)];
reshape(pi_lb_to_reshape, n_supp, n_supp)

pi_ppd = [0.5708,    0.0000,    0.0000,    0.0000,    0.0709,    0.2404,    0.0000,    0.0000,    0.0000,    0.0450,    0.0265, ...
   0.0000,    0.0000,    0.0000,    0.0202]';
pi_ppd_to_reshape = [pi_ppd; 1 - sum(pi_ppd)];
reshape(pi_ppd_to_reshape, n_supp, n_supp)
    
gamma_pmf =  [0.5708,    0.0000,    0.0000,    0.0000,    0.0709,    0.2404,    0.0000,    0.0000,    0.0000,    0.0450, ...
        0.0265,    0.0000,    0.0000,    0.0000,    0.0202]';
gamma_pmf_to_reshape = [gamma_pmf; 1 - sum(gamma_pmf)];
reshape(gamma_pmf_to_reshape, n_supp, n_supp)
        
lambda = [0     1     1     1;
             0     0     1     1;
             0     0     0     1;
             0     0     0     0];
         
 
beta = 0.1735;
    
        
theta_0 = [beta; pi; upsilon; gamma_pmf; lambda(:)];
[m_ineq, m_eq, J1, J2, m_eq_std, m_ineq_std] = compute_moments_stdev(theta_0, y_supp, n_supp, d, p_a, p_e, rho_l, 0);
m_eq


theta_feas = [theta_0';
               [beta_lb; pi_lb; upsilon; gamma_pmf; lambda(:)]';
               [beta_ppd; pi_ppd; upsilon; gamma_pmf; lambda(:)]'];

LB_theta = [-3; zeros(n_supp^2 - 1, 1); zeros(n_supp - 1, 1); zeros(n_supp^2 - 1, 1); zeros(n_supp^2, 1)];
UB_theta = [3; ones(n_supp^2 - 1, 1); ones(n_supp - 1, 1); ones(n_supp^2 - 1, 1); ones(n_supp^2, 1)];

% Require the PMF parameters to sum to 1 or less
A_theta = [ 0, ones(1, n_supp^2 - 1), zeros(1, n_supp - 1), zeros(1, n_supp^2 - 1), zeros(1, n_supp^2) ;
            0, zeros(1, n_supp^2 - 1), ones(1, n_supp - 1), zeros(1, n_supp^2 - 1), zeros(1, n_supp^2) ;
            0, zeros(1, n_supp^2 - 1), zeros(1, n_supp - 1), ones(1, n_supp^2 - 1), zeros(1, n_supp^2) ];
        
b_theta = [1;
           1;
           1;];

[m_eq, m_ineq, m_eq_std, m_ineq_std] = compute_moments_stdev(theta_0, y_supp, n_supp, d, p_a, p_e, rho_l, 1);
                                                                                                                      
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
%KMSoptions.FeasAll = 1; % try this on the cluster. locally: check how beta is working
KMSoptions.parallel = 1;

[KMS_confidence_interval,KMS_output] = KMS_0_Main(d, theta_0, y_supp, n_supp, p_a, p_e, rho_l, p, [], LB_theta, UB_theta, A_theta, b_theta, 0.1, 'two-sided', 'AS' , NaN, NaN, [], KMSoptions);
                                                
% diagnostics
KMS_output


disp('LOWER')

% lower bound
theta = KMS_output.thetaL_EAM'

[m_ineq, m_eq, J1, J2, m_eq_std, m_ineq_std] = compute_moments_stdev(theta, y_supp, n_supp, d, p_a, p_e, rho_l, 0);
m_eq

% code copied from compute_moments_stdev
beta = theta(1)

pi = theta((1+1):(n_supp^2));
pi = [pi; 1 - sum(pi)];
pi = reshape(pi, n_supp, n_supp)

upsilon = theta((n_supp^2 + 1):(n_supp^2 + n_supp - 1));
upsilon = [upsilon; 1 - sum(upsilon)]

gamma_pmf = theta((n_supp^2 + n_supp):(n_supp^2 + n_supp + n_supp^2 - 2));
gamma_pmf = reshape([gamma_pmf; 1 - sum(gamma_pmf)], ...
                    n_supp, n_supp)

gamma = zeros(n_supp, n_supp);
for i = 1:n_supp
    for j = 1:n_supp
        gamma(i,j) = sum(gamma_pmf(1:i, 1:j), 'all');
    end
end          

lambda = reshape(theta((n_supp^2 + n_supp + n_supp^2 - 1):end), ...
                    n_supp, n_supp)
% display results
gamma

disp('UPPER')


theta = KMS_output.thetaU_EAM'

[m_ineq, m_eq, J1, J2, m_eq_std, m_ineq_std] = compute_moments_stdev(theta, y_supp, n_supp, d, p_a, p_e, rho_l, 0);
m_eq


% code copied from compute_moments_stdev
beta = theta(1)

pi = theta((1+1):(n_supp^2));
pi = [pi; 1 - sum(pi)];
pi = reshape(pi, n_supp, n_supp)

upsilon = theta((n_supp^2 + 1):(n_supp^2 + n_supp - 1));
upsilon = [upsilon; 1 - sum(upsilon)]

gamma_pmf = theta((n_supp^2 + n_supp):(n_supp^2 + n_supp + n_supp^2 - 2));
gamma_pmf = reshape([gamma_pmf; 1 - sum(gamma_pmf)], ...
                    n_supp, n_supp)

gamma = zeros(n_supp, n_supp);
for i = 1:n_supp
    for j = 1:n_supp
        gamma(i,j) = sum(gamma_pmf(1:i, 1:j), 'all');
    end
end          

lambda = reshape(theta((n_supp^2 + n_supp + n_supp^2 - 1):end), ...
                    n_supp, n_supp)
% display results
gamma




% % check beta formula
% 
% y_dummies = dummyvar(categorical(d.max_postmath_std));
% 
% n = height(d);
% y_cdf_dummies = repmat(d.max_postmath_std, 1, 4) <= repmat(y_supp', n, 1);
% y_lt_dummies = repmat(d.max_postmath_std, 1, 4) < repmat(y_supp', n, 1);
% 
% mumbai_for_dummies = repmat(d.mumbai == 1, 1, n_supp);
% vado_t_for_dummies = repmat(d.mumbai == 0 & d.bal == 1, 1, n_supp);
% vado_c_for_dummies = repmat(d.mumbai == 0 & d.bal == 0, 1, n_supp);
% 
% beta_trans = - y_supp + sum(repmat(y_supp', n_supp, 1) ./ repmat(upsilon, 1, n_supp) .* pi, 2);
% beta_applied = (y_dummies .* mumbai_for_dummies) * beta_trans;
% 
% beta_mom = mean(beta_applied) - beta * p_a