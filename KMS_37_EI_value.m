function [EI] = KMS_37_EI_value(theta,q,theta_hash,y_supp, n_supp, n_x_supp, d, p_a, p_e, rho_l, bs_classyears, dmodel,KMSoptions)
%% Code description: Expected Improvement with fmincon
% This function computes the expected improvement at theta.
%
% INPUT:
% theta         (dim_p)-by-1 parameter vector
%
% q             dim_p-by-1 directional vector.  This is either  p or -p.
%
% dmodel        DACE kriging model (computed using theta_A)
%
% KMSoptions    This is a structure of additional inputs held 
%               constant over the program.  In the 2x2 entry game, 
%               KMSoptions includes the support for the covariates
%               and the probability of support point occuring. 
%  
% OUTPUT:
%   Ei          J-by-1 vector of expected improvements for each moment 
%               inequality minus gamma.  

%% Extract relevant information from KMSoptions
J1          = KMSoptions.J1;
J2          = KMSoptions.J2;
n           = KMSoptions.n;

%% Extract relevant information for BCS_EAM from KMSoptions
BCS_EAM = KMSoptions.BCS_EAM;
component = KMSoptions.component;

%% Expected Improvement
% We compute expected improvement for each moment inequality j=1...J:
%   EI_j = (q'theta - q'theta_#)_{+}*Phi( (h_j(theta) - c(theta))/s(theta))
% where h_j(theta) is the standardized moment
%   h_j(theta) = sqrt(n)*m_j(X,theta)/sigma(X)
% and c_L(theta), s_L(theta) are from the DACE auxillary model.
% Note that we are searching over the space of theta such that 
%   q'theta >= q'theta_#
% so the max(0,.) is not required.

% Step 1) h_j(theta)
% We compute the standardized moments
% Theoretical momoments

[m_ineq, m_eq, J1, J2, m_eq_std, m_ineq_std] = compute_moments_stdev(theta, y_supp, n_supp, d, p_a, p_e, rho_l, 1, n_x_supp);

% Standardized momoments
h_theta = sqrt(n)*(([m_ineq ; m_eq])./[m_ineq_std; m_eq_std]);

% Step 2) c(theta) and s(theta)
% Approximated value of c(theta) using DACE
if BCS_EAM == 1
	lambda = theta(component);
	c_theta    = predictor(lambda,dmodel);
	[~,~,mse,~]= predictor(lambda,dmodel);
else
	c_theta    = predictor(theta,dmodel);
	[~,~,mse,~]= predictor(theta,dmodel);
end
% Compute s(theta) 
s = sqrt(mse);

% Step 3) Compute expected improvement minus gamma
EI = q.'*(theta - theta_hash)*(-normcdf(-(h_theta-c_theta)/s));

end
