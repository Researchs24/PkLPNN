function dz = pklpnnNew(t,z,phi,r,l1bound,k)%,seuil,tseuil)
%  PKLPNNNEW Implements the dynamics $\frac{dx}{dt}$ and $\frac{d\lambda}{dx}$. ver: 1.2
% 
% dz = lpnn(t,z,phi,r,l1bound,k)is used within the ODE solver. The needed parameters 
% are: - phi: a (m x n) measurement matrix - r: a (m x 1) vector of the observed 
% signal - l1bound: a number. It is equal to eta. - k: a number. This function 
% applies T(x) to the inputs
% 
% Smoothing function $|\alpha|_k = \sqrt{\alpha^2+\frac{1}{k^2}}$, $\alpha \in 
% R$
% 
% Obj. fct $\frac{1}{2}\|r-\phi x\|^2_2$ s.t. $\|x\|_{1,k} \leq \eta$
% 
% $$\|x\|_{k,1}=\sum_{i=1}^n|x_i|_k$$
% 
% $$|x_i|_{k,1}=\sqrt{x_i^2+\frac{1}{k^2}}$$
% 
% $$\frac{dx}{dt}=\phi^t(r-\phi x(t))-\lambda(t)\left(\frac{x(t)_i}{\sqrt{\left(x(t)^2_i+\frac{1}{k^2}\right)}} 
% \right)^t_{i=1..n} = =\phi^t(r-\phi x(t))-\lambda(t)\left(\frac{x(t)_i}{|x_i|_{k,1}} 
% \right)^t_{i=1..n}$$
% 
% $$\frac{d\lambda}{dt}=\nabla_\lambda L = \sum^n_{i=1} \sqrt{x(t)_i^2 + \frac{1}{k^2}} 
% - \eta = \|x\|_{k,1}-\eta$$
% 
% This function is used by the ODE solver to compute the solution System dynamics
    n = length(z) - 1;                              % Signal length
    
    db = phi'*(r - phi*z(1:n));
    dz = db -z(n+1)*(z(1:n)./sqrt((z(1:n).^2+(1/k^2))));  % Variable neurones (n)
    dz(n+1) = k1metrix2(z(1:n), k) - l1bound;       % Lagrange neurone  (1)
%dz = dz';
end
%% 
% This function computes $||x||_{k,1}=\sum_{i=1}^n|x_i|_k$ with $|x_i|_{k,1}=\sqrt{x_i^2+\frac{1}{k^2}}$
function mt = k1metrix2(x, k)
    f = @(px, pk) sqrt(px.^2 + 1/pk^2);
    
    mt = sum(f(x, k));
end