function dz = pklpnn1(t,z,phi,r,l1bound,k)%,seuil,tseuil)
%  PKLPNN1 implements the dynamics $\frac{dx}{dt}$ and $\frac{d\lambda}{dt}$. ver: 1.2
% 
% dz = lpnn(t,z,phi,r,l1bound,k)is used within the ODE solver.
% 
% The needed parameters are: 
%% 
% * $\phi \in R^{m \times n}$ the measurement matrix
% * $r \in R^{m \times 1}$ vector of the observed signal
% * l1boun is $\eta$
% * k  a number
%% 
% $$$\frac{dx}{dt}=\phi^t \left(r-\phi x\right)-\lambda \ldotp \nabla_x \left\|x{\left\|\right.}_{k,1} 
% \right.$$
% 
% $$$\frac{d\lambda }{dt}=\left\|x{\left\|\right.}_{k,1} -\eta \right.$$
% 
% $$$\nabla_x \left\|x\right\|_{k,1} =\frac{1}{1+e^{-\textrm{kx}} }-\frac{1}{1+e^{\textrm{kx}} 
% }$$ 
% 
% This function is used by the ODE solver to compute the solution System dynamics
n = length(z) - 1;                          % Signal length
f = @(sx) 1./(1+exp(-sx)) - 1./(1+exp(sx)); % The function expression
db = phi'*(r - phi*z(1:n));
dz(1:n) = db - z(n+1) * f(k*z(1:n));        % Variable neurones (n)
dz(n+1) = k1metrix(z(1:n), k) - l1bound;    % Lagrange neurone  (1)
dz = dz';