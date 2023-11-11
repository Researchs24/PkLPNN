function dz = pklpnn1(t,z,phi,r,l1bound,k)%,seuil,tseuil)
%  PKLPNN1 implements the dynamics dx/dt and dlambda/dt. ver: 1.2
% 
% dz = lpnn(t,z,phi,r,l1bound,k)is used within the ODE solver. The needed parameters 
% are: - phi: a (m x n) measurement matrix - r: a (m x 1) vector of the observed 
% signal - l1bound: a number. It is equal to eta. - k: a number. This function 
% applies T(x) to the inputs
% 
% $$\frac{\textrm{dx}}{\textrm{dt}}=\phi^t \left(r-\phi x\right)-\lambda \ldotp 
% \nabla_x \left\|x{\left\|\right.}_{k,1} \right.$$                           
% $$\frac{d\lambda }{\textrm{dt}}=\left\|x{\left\|\right.}_{k,1} -\eta \right.$$              
% $$\nabla_x \left\|x{\left\|\right.}_{k,1} =\frac{1}{1+e^{-\textrm{kx}} }-\frac{1}{1+e^{\textrm{kx}} 
% }\right.$$
% This function is used by the ODE solver to compute the solution
% System dynamics
n = length(z) - 1;                          % Signal length
f = @(sx) 1./(1+exp(-sx)) - 1./(1+exp(sx)); % The function expression
db = phi'*(r - phi*z(1:n));
dz(1:n) = db - z(n+1) * f(k*z(1:n));        % Variable neurones (n)
dz(n+1) = k1metrix(z(1:n), k) - l1bound;    % Lagrange neurone  (1)
dz = dz';