function [m_ineq, m_eq, J1, J2, m_eq_std, m_ineq_std] = compute_moments_stdev(theta, y_supp, n_supp, d, p_a, p_e, rho_l, compute_stdev, n_x_supp)
% Compute values and standard deviations of the moments

    beta = theta(1);
    
    % TODO: note that xi needs a polytope restriction
    xi = theta(2:n_x_supp);
    xi = [xi; (1 - sum(xi))];
    
    conditional_params = theta((n_x_supp + 1):end);
    
    % data
        
    y_dummies = dummyvar(categorical(d.max_postmath_std));
    x_dummies = dummyvar(categorical(d.max_premath_std));

    n = height(d);
    
    y_cdf_dummies = repmat(d.max_postmath_std, 1, 4) <= repmat(y_supp', n, 1);
    
    y_lt_dummies = repmat(d.max_postmath_std, 1, 4) < repmat(y_supp', n, 1);

    mumbai_for_dummies = repmat(d.mumbai == 1, 1, n_supp);
    
    vado_t_for_dummies = repmat(d.mumbai == 0 & d.bal == 1, 1, n_supp);
    
    vado_c_for_dummies = repmat(d.mumbai == 0 & d.bal == 0, 1, n_supp);
   
    
    xi_applied = repmat((1 - d.mumbai == 1), 1, n_x_supp) .* x_dummies;
    xi_moment = mean(xi_applied)' - xi * (1 - p_a);
    
    m_eq = [xi_moment; -xi_moment];
    
    if compute_stdev
        m_eq_std = [std(xi_applied)'; std(xi_applied)'];
    else
        m_eq_std = [];
    end
    
    
    % separate x-specific parameters
    conditional_params = reshape(conditional_params, length(conditional_params)/n_x_supp, n_x_supp)';
    conditional_params = [zeros(n_x_supp, 1) conditional_params];
    
    
    beta_applied = zeros(n, 1);
    
    J2 = length(xi_moment) + 1; % includes beta
    
    m_ineq = [];
    m_ineq_std = [];
    
    
    for l = 1:n_x_supp
        
        % x probability
        xi_l = xi(l);
        x_dummy = x_dummies(:,l);
        x_for_dummies = repmat(x_dummy, 1, n_supp);
        
        % conditional-on-x parameters
        x_specific_params = conditional_params(l,:)';

        pi = x_specific_params((1+1):(n_supp^2));
        pi = [pi; 1 - sum(pi)];
        pi = reshape(pi, n_supp, n_supp);

        upsilon = x_specific_params((n_supp^2 + 1):(n_supp^2 + n_supp - 1));
        upsilon = [upsilon; 1 - sum(upsilon)];

        gamma_pmf = x_specific_params((n_supp^2 + n_supp):(n_supp^2 + n_supp + n_supp^2 - 2));
        gamma_pmf = reshape([gamma_pmf; 1 - sum(gamma_pmf)], ...
                            n_supp, n_supp);

        gamma = zeros(n_supp, n_supp);
        for j = 1:n_supp
            for k = 1:n_supp
                gamma(j,k) = sum(gamma_pmf(1:j, 1:k), 'all');
            end
        end          

        lambda = reshape(x_specific_params((n_supp^2 + n_supp + n_supp^2 - 1):end), ...
                            n_supp, n_supp);
                        
        % beta contribution
        beta_trans = - y_supp + sum(repmat(y_supp', n_supp, 1) ./ repmat(upsilon, 1, n_supp) .* pi, 2);
        beta_applied = beta_applied + (y_dummies .* mumbai_for_dummies .* repmat(x_dummy, 1, n_supp ) ) * beta_trans;
        
       
        % experimental marginals
        c_e_marginal_mom = xi_l .* (1 - p_a) .* (1 - p_e) .* sum(pi, 2) - mean(y_dummies .* vado_c_for_dummies .* x_dummy)';

        t_e_marginal_mom = xi_l .* ( (1 - p_a) .* p_e .* sum(pi, 1))' - mean(y_dummies .* vado_t_for_dummies .* x_dummy)';

        upsilon_mom = xi_l .* (1 - p_a) .* (1 - p_e) .* upsilon - mean(y_dummies .* vado_c_for_dummies .* x_dummy)';

        % gamma moment equalities
        scaled_lambda = lambda ./ ( xi_l .* (1 - p_a) .* (1 - p_e));
        scaled_1_minus_lambda = (1 - lambda) ./ ( xi_l .* (1 - p_a) .* p_e);

        lambda_applied = repmat(y_cdf_dummies .* vado_c_for_dummies .* x_for_dummies, 1, n_supp) .* ...
                                repmat(scaled_lambda(:)', n, 1) ...
                            + (kron(y_cdf_dummies .* vado_t_for_dummies .* x_for_dummies, ones(1, n_supp)) .* ...
                                repmat(scaled_1_minus_lambda(:)', n, 1) );
        gamma_mom_eq = mean(lambda_applied)' - gamma(:);

        % gamma moment inequalities

        scaled_gamma_untreat = (gamma .* (xi_l .* (1 - p_a) .* (1 - p_e)));
        gamma_c_e_mom_ineq = mean(repmat(scaled_gamma_untreat(:)', n, 1) ...
                                    - repmat(y_cdf_dummies .* vado_c_for_dummies .* x_for_dummies, 1, n_supp))';

        scaled_gamma_treat = gamma .* (xi_l .* (1 - p_a) .* p_e);
        gamma_t_e_mom_ineq = mean( repmat( scaled_gamma_treat(:)', n, 1)  ...
                                    - kron(y_cdf_dummies .* vado_t_for_dummies .* x_for_dummies, ones(1, n_supp)) )';


        % dependence constraint
        gamma_extended = [zeros(n_supp + 1, 1) [zeros(1, n_supp); gamma]];
        gamma_sum = gamma + gamma_extended(1:n_supp, 1:n_supp) + ...
                gamma_extended(1:n_supp, 2:(n_supp + 1)) + gamma_extended(2:(n_supp + 1), 1:n_supp);

        max_spear_coefs = rho_l .* 3 ./(xi_l .* (1 - p_a) .* p_e) .* repmat(upsilon, n_supp, 1) .* ...
            gamma_sum(:);

        max_spear_applied = kron(y_dummies .* vado_t_for_dummies .* x_for_dummies, ones(1, n_supp)) * max_spear_coefs;

        upsilon_mat = repmat(upsilon, 1, n_supp);
        upsilon_cum = [0; cumsum(upsilon)];
        upsilon_cum_mat = repmat(upsilon_cum(1:n_supp), 1, n_supp);

        upsilon_sum = upsilon_mat + 2 .* upsilon_cum_mat - 1;

        spear_pi_coefs = 3 .* pi(:) .* upsilon_sum(:) ./ (xi_l .* (1-p_a) .* p_e) ;

        spear_pi_applied = kron(x_for_dummies .* vado_t_for_dummies .* (y_dummies + 2 .*  y_lt_dummies - 1), ones(1, n_supp)) * spear_pi_coefs;

        spear_applied = max_spear_applied - spear_pi_applied;

        spear_mom_ineq = mean(spear_applied) - rho_l * 3;

        m_ineq = [m_ineq; gamma_c_e_mom_ineq; gamma_t_e_mom_ineq; spear_mom_ineq];
        
        m_eq_l = [c_e_marginal_mom; t_e_marginal_mom; upsilon_mom; gamma_mom_eq];
        J2 = J2 + length(m_eq_l);
        % treating moment equalities as 2 opposing moment inequalities
        m_eq_l = [m_eq_l; -m_eq_l];
        m_eq = [m_eq; m_eq_l];

        if compute_stdev
            c_e_marginal_std = std(y_dummies .* vado_c_for_dummies .* x_for_dummies)';
            t_e_marginal_std = std(y_dummies .* vado_t_for_dummies .* x_for_dummies)';
            upsilon_std = std(y_dummies .* vado_c_for_dummies .* x_for_dummies)';
            gamma_mom_eq_std = std(lambda_applied)';

            % inequalities
            gamma_c_e_mom_ineq_std = std(repmat(y_cdf_dummies .* vado_c_for_dummies .* x_for_dummies, 1, n_supp))';
            gamma_t_e_mom_ineq_std = std(kron(y_cdf_dummies .* vado_t_for_dummies .* x_for_dummies, ones(1, n_supp)))';
            spear_mom_ineq_std = std(spear_applied);

            % aggregate
            m_eq_std_l = [c_e_marginal_std; t_e_marginal_std; upsilon_std; gamma_mom_eq_std];
            % treating moment equalities as 2 opposing moment inequalities
            m_eq_std_l = [m_eq_std_l; m_eq_std_l];
            m_eq_std = [m_eq_std; m_eq_std_l];
            m_ineq_std = [m_ineq_std; gamma_c_e_mom_ineq_std; gamma_t_e_mom_ineq_std; spear_mom_ineq_std];
        else
            continue;
        end
        
    end
    
    beta_mom = mean(beta_applied) - beta * p_a;
    m_eq = [beta_mom; -beta_mom; m_eq];
   
    J1 = length(m_ineq);
    
    beta_std = std(beta_applied);
    m_eq_std = [beta_std; beta_std; m_eq_std];
    
end 