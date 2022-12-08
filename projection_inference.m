clear

seed = 47;

rng(seed);



d = readtable("/Users/mdg5396/Data/external/balsakhi/nlp/input/full_dataset.csv");
d = d(:, ["mumbai" "bal" "max_premath_std" "max_postmath_std" "classyearid"]);

% omit mumbai treated group
d = d(d.mumbai == 0 | (d.mumbai == 1 & d.bal == 0), :);

rho_l = 0.9;
B = 100;
alpha = 0.1;

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

% feasible (optimal) value of parameters
beta = 0.1735;

upsilon = tab_max_postmath_std(1:(n_supp -1),3) * 0.01;

pi = [0.5636,    0.0331,    0.0450,    0.0000 , ...
        0.0072,    0.2782,    0.0000,    0.0000 , ...
        0.0000,    0.0000,    0.0003,    0.0464 , ...
        0.0000,    0.0000,    0.0261    ]';
    
gamma = [0.5708    0.6418    0.6418    0.6418;
            0.5708    0.8822    0.9272    0.9272;
            0.5708    0.8822    0.9536    0.9739;
            0.5708    0.8822    0.9536    1.0000];
gamma_pmf =     [0.5708,    0.0709,    0.0000,    0.0000, ...
                    0.0000,    0.2404,    0.0450,    0.0000, ...
                    0.0000,    0.0000,    0.0265,    0.0202, ...
                    0.0000,    0.0000,    0.0000]'   
        
lambda = [0     1     1     1;
             0     0     1     1;
             0     0     0     1;
             0     0     0     0];
         
         
theta_0 = [beta; pi; upsilon; vec(gamma); vec(lambda)];

LB_theta = [-3; zeros(n_supp^2 - 1, 1); zeros(n_supp - 1, 1); zeros(n_supp^2 - 1, 1); zeros(n_supp^2 - 1, 1)];
UB_theta = [3; ones(n_supp^2 - 1, 1); ones(n_supp - 1, 1); ones(n_supp^2 - 1, 1); ones(n_supp^2 - 1, 1)];



[m_eq, m_ineq, m_eq_std, m_ineq_std] = compute_moments_stdev(theta_0, y_supp, n_supp, d, p_a, p_e, rho_l, 1);
                                                                                                                      
p = [1; zeros(length(theta_0) - 1, 1)];


KMSoptions.B            = B;
KMSoptions.parallel     = 1 ;   % Specify if using parallel computing.  Default is to set parallel = 1 if user has toolbox installed and parallel not specified.
KMSoptions.num_cores    = [];   % Number of cores if using parallel.  If not specified, use max number of cores.
KMSoptions.seed         = 0;    % Seed value
KMSoptions.CVXGEN       = 1;    % Set equal to 1 if CVXGEN is used.  Set equal to 0 if CVX is used


[KMS_confidence_interval,KMS_output] = KMS_0_Main(d, theta_0, ...
            p, [], LB_theta, UB_theta, [], [], 0.1, 'two-sided', 'AS' , NaN, NaN, [], KMSoptions);
                                                            

% next: bootstrap moments only

% draw boostrap samples
classyears = unique(d.classyearid);
bs_classyear_indices = randi(length(classyears), B);
bs_classyears = classyears(bs_classyear_indices);

m_eq_bs = zeros(length(m_eq), B);
m_ineq_bs = zeros(length(m_ineq), B);

parfor b = 1:B
    classyearid = bs_classyears(:,b);
    d_b = innerjoin(table(classyearid), d);
    
    
    [m_eq_b, m_ineq_b, m_eq_std_b, m_ineq_std_b] = compute_moments_stdev(y_supp, n_supp, d_b, p_a, p_e, rho_l, ...
                                                                beta, pi, upsilon, gamma, lambda, 0);
    
    m_eq_bs(:,b) = m_eq_b;
    m_ineq_bs(:,b) = m_ineq_b;
end

% recenter bootstrap moments
G_eq   = sqrt(n).*(m_eq_bs - repelem(m_eq, 1, B))./repmat(m_eq_std, 1, B);
G_ineq = sqrt(n).*(m_ineq_bs - repmat(m_ineq, 1, B))./repmat(m_ineq_std, 1, B);

% Note: moment is m(W,theta) = f(W) + g(theta)
% So measure of "close to binding" is given by
% xi = (1/kappa)*sqrt(n)*(f(W) + g(theta))/(stdev)


% Measure of close to binding (Equation 2.8)
% for the moment inequalities, and zero for the moment equalities
xi_ineq = (1/kappa) * sqrt(n) .* m_ineq ./m_ineq_std;
xi = [xi_ineq; zeros(length(m_eq),1)];

% GMS function (Equation 2.9)
% phi_test is the GMS function evaluated at the measure of "close
% to binding" xi, computed above.
% Following KMS, the hard-threshing GMS function is used,
% where phi = -inifity if xi < -1, 0 else.
phi_test = phi(xi);

G = [G_ineq;G_eq];
S_3_boot = max(0, max(G + repmat(phi_test, 1, B)));
c_val = quantile(S_3_boot, 1 - alpha)

% constraint violation
m_theta = sqrt(n)*(([m_eq; m_ineq])./[m_eq_std; m_ineq_std]);
sum(max(0,m_theta - c_val).^2)
