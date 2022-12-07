clear

rng(47);



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

upsilon = tab_max_postmath_std(:,3) * 0.01;

pi = [0.5636    0.0331    0.0450    0.0000;
        0.0072    0.2782    0.0000    0.0000;
        0.0000    0.0000    0.0003    0.0464;
        0.0000    0.0000    0.0261    0.0000];
    
gamma = [0.5708    0.6418    0.6418    0.6418;
            0.5708    0.8822    0.9272    0.9272;
            0.5708    0.8822    0.9536    0.9739;
            0.5708    0.8822    0.9536    1.0000];
        
lambda = [0     1     1     1;
             0     0     1     1;
             0     0     0     1;
             0     0     0     0];
         
         
theta_0 = [beta; vec(pi); upsilon; vec(gamma); vec(lambda)]

u_beta = theta_0(1);
u_pi = reshape(theta_0((1+1):(1+n_supp^2)), ...
                n_supp, n_supp)
pi
u_upsilon = theta_0((1 + n_supp^2 + 1):(1 + n_supp^2 + n_supp))
upsilon
u_gamma = reshape(theta_0((1 + n_supp^2 + n_supp + 1):(1 + n_supp^2 + n_supp + n_supp^2)), ...
                    n_supp, n_supp)
gamma
u_lambda = reshape(theta_0((1 + n_supp^2 + n_supp + n_supp^2 + 1):(1 + n_supp^2 + n_supp + n_supp^2 + n_supp^2)), ...
                    n_supp, n_supp)
lambda


%   beta:           scalar object of interest
%   pi:             JxJ matrix of the joint distribution of potential outcomes in $e$
%   upsilon:        J-vector representing the PMF of Y in the untreated group in $e$
%   gamma:          JxJ matrix of $min \{ P^e(Y \leq y_{j} | T = 0), P^e(Y \leq y_{j} | T = 1) \}$.
%                   For the purposes of calculating the Spearman constraint, the matrix should be 
%                   augmented by an initial row and column of zeros.
%   lambda:         JxJ matrix of weights $\in \{0,1\}$ denoting whether $P^e(Y \leq y_{j} | T = 0) \leq P^e(Y \leq y_{j} | T = 1) \}$



[m_eq, m_ineq, m_eq_std, m_ineq_std] = compute_moments_stdev(y_supp, n_supp, d, p_a, p_e, rho_l, ...
                                                                beta, pi, upsilon, gamma, lambda, 1);
                                                                                                                      
p = [1; zeros(length(theta_0) - 1, 1)];

                                                            
[KMS_confidence_interval,KMS_output] = KMS_0_Main(d, beta, pi, upsilon, gamma, lambda, ...
            p, [], LB_theta,UB_theta, A_theta,b_theta,alpha,type,method,kappa,phi,CVXGEN_name,KMSoptions_app);
                                                            

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
