function [m_ineq, m_eq, J1, J2, m_eq_std, m_ineq_std] = compute_moments_stdev(theta, y_supp, n_supp, d, p_a, p_e, rho_l, compute_stdev)
% Compute values and standard deviations of the moments

    % setup - parameters
    beta = theta(1);
    
    pi = theta((1+1):(n_supp^2));
    pi = [pi; 1 - sum(pi)];
    pi = reshape(pi, n_supp, n_supp);
                
    upsilon = theta((n_supp^2 + 1):(n_supp^2 + n_supp - 1));
    upsilon = [upsilon; 1 - sum(upsilon)];
    
    gamma_pmf = theta((n_supp^2 + n_supp):(n_supp^2 + n_supp + n_supp^2 - 2));
    gamma_pmf = reshape([gamma_pmf; 1 - sum(gamma_pmf)], ...
                        n_supp, n_supp);
                    
    gamma = zeros(n_supp, n_supp);
    for i = 1:n_supp
        for j = 1:n_supp
            gamma(i,j) = sum(gamma_pmf(1:i, 1:j), 'all');
        end
    end          
    
    lambda = reshape(theta((n_supp^2 + n_supp + n_supp^2 - 1):end), ...
                        n_supp, n_supp);

    % data
    y_dummies = dummyvar(categorical(d.max_postmath_std));
    
    n = height(d);
    y_cdf_dummies = repmat(d.max_postmath_std, 1, 4) <= repmat(y_supp', n, 1);
    y_lt_dummies = repmat(d.max_postmath_std, 1, 4) < repmat(y_supp', n, 1);
    
    mumbai_for_dummies = repmat(d.mumbai == 1, 1, n_supp);
    vado_t_for_dummies = repmat(d.mumbai == 0 & d.bal == 1, 1, n_supp);
    vado_c_for_dummies = repmat(d.mumbai == 0 & d.bal == 0, 1, n_supp);

    % beta
    
    beta_trans = - y_supp + sum(repmat(y_supp', n_supp, 1) ./ repmat(upsilon, 1, n_supp) .* pi, 2);
    
    beta_applied = (y_dummies .* mumbai_for_dummies) * beta_trans;
    
    beta_mom = mean(beta_applied) - beta * p_a;
   
    % experimental marginals
    c_e_marginal_mom = (1 - p_a) .* (1 - p_e) .* sum(pi, 2) - mean(y_dummies .* vado_c_for_dummies)';
    
    t_e_marginal_mom = ( (1 - p_a) .* p_e .* sum(pi, 1))' - mean(y_dummies .* vado_t_for_dummies)';
    
    upsilon_mom = (1 - p_a) .* (1 - p_e) .* upsilon - mean(y_dummies .* vado_c_for_dummies)';
    
    % gamma moment equalities
    lambda_applied = repmat(y_cdf_dummies .* vado_c_for_dummies, 1, n_supp) .* ...
                            repmat(vec((lambda ./ ((1 - p_a) .* (1 - p_e))))', n, 1) ...
                        + (kron(y_cdf_dummies .* vado_t_for_dummies, ones(1, n_supp)) .* ...
                            repmat(vec((1 - lambda) ./ ((1 - p_a) .* p_e))', n, 1) );
    gamma_mom_eq = mean(lambda_applied)' - vec(gamma);
    
    % gamma moment inequalities
    lambda_applied = repmat(y_cdf_dummies .* vado_c_for_dummies, 1, n_supp) .* ...
                        repmat(vec((lambda ./ ((1 - p_a) .* (1 - p_e))))', n, 1) ...
                    + (kron(y_cdf_dummies .* vado_t_for_dummies, ones(1, n_supp)) .* ...
                        repmat(vec((1 - lambda) ./ ((1 - p_a) .* p_e))', n, 1) );

    gamma_c_e_mom_ineq = mean(repmat(vec((gamma .* ((1 - p_a) .* (1 - p_e))))', n, 1) ...
                                - repmat(y_cdf_dummies .* vado_c_for_dummies, 1, n_supp))';
    
    gamma_t_e_mom_ineq = mean( repmat( vec( gamma .* ((1 - p_a) .* p_e))', n, 1)  ...
                                - kron(y_cdf_dummies .* vado_t_for_dummies, ones(1, n_supp)) )';
    
    
    % dependence constraint
    gamma_extended = [zeros(n_supp + 1, 1) [zeros(1, n_supp); gamma]];
    
    max_spear_coefs = rho_l .* 3 ./((1 - p_a)*p_e) .* repmat(upsilon, n_supp, 1) .* ...
        vec(gamma + gamma_extended(1:n_supp, 1:n_supp) + ...
            gamma_extended(1:n_supp, 2:(n_supp + 1)) + gamma_extended(2:(n_supp + 1), 1:n_supp));
    
    max_spear_applied = kron(y_dummies .* vado_t_for_dummies, ones(1, n_supp)) * max_spear_coefs;
    
    upsilon_mat = repmat(upsilon, 1, n_supp);
    upsilon_cum = [0; cumsum(upsilon)];
    upsilon_cum_mat = repmat(upsilon_cum(1:n_supp), 1, n_supp);
    
    spear_pi_coefs = 3 .* vec(pi) .* vec(upsilon_mat + 2 .* upsilon_cum_mat - 1) ./ ((1-p_a) .* p_e) ;
    
    spear_pi_applied = kron(vado_t_for_dummies .* (y_dummies + 2 .*  y_lt_dummies - 1), ones(1, n_supp)) * spear_pi_coefs;
    
    spear_applied = max_spear_applied - spear_pi_applied;
    
    spear_mom_ineq = mean(spear_applied) - rho_l * 3;
       
    m_ineq = [gamma_c_e_mom_ineq; gamma_t_e_mom_ineq; spear_mom_ineq];
    J1 = length(m_ineq);
     
    m_eq = [beta_mom; c_e_marginal_mom; t_e_marginal_mom; upsilon_mom; gamma_mom_eq];
    J2 = length(m_eq);
    
    if compute_stdev
        % equalities
        beta_std = std(beta_applied);
        c_e_marginal_std = std(y_dummies .* vado_c_for_dummies)';
        t_e_marginal_std = std(y_dummies .* vado_t_for_dummies)';
        upsilon_std = std(y_dummies .* vado_c_for_dummies)';
        gamma_mom_eq_std = std(lambda_applied)';
  
        % inequalities
        gamma_c_e_mom_ineq_std = std(repmat(y_cdf_dummies .* vado_c_for_dummies, 1, n_supp))';
        gamma_t_e_mom_ineq_std = std(kron(y_cdf_dummies .* vado_t_for_dummies, ones(1, n_supp)))';
        spear_mom_ineq_std = std(spear_applied);
        
        % aggregate
        m_eq_std = [beta_std; c_e_marginal_std; t_e_marginal_std; upsilon_std; gamma_mom_eq_std];
        m_ineq_std = [gamma_c_e_mom_ineq_std; gamma_t_e_mom_ineq_std; spear_mom_ineq_std];
    else
        m_eq_std = [];
        m_ineq_std = [];
    end
    
    
end 