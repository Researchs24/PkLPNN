function dz = lassolpnn(t,u,phi,r,l1bound)
%  LASSOLPNN $\frac{\mathrm{du}}{\mathrm{dt}}={2\phi }^t \left(r-\phi x\right)-\lambda
% 
% \left(u-x\right)$(37a)  $\frac{d\lambda }{\textrm{dt}}=\left\|x{\left\|\right.}_1 
% -\eta \right.$(37b)  $x_i =T_{\lambda } \left(u_i \right)=\left\lbrace \begin{array}{cc}0 
% & \left|u_i \right|\le \lambda \\u_i -\lambda \textrm{sign}\left(u_i \right) 
% & \left|u_i \right|>\lambda \end{array}\right.$(35) This function is used by 
% the ODE solver to compute the solution System dynamics dx/dt = phi'*(r - phi*x) 
% - lambda*grad_x||x||_{k,1} d f = @(sx) 1./(1+exp(-sx)) - 1./(1+exp(sx)); % The 
% function expression TODO: Add the theoritical logic behind these expressions
n = length(u) - 1;                          % Signal length
ind = abs(u(1:n)) > u(n+1);                 % x = T(u)
x = zeros(n,1);
x(ind) = u(ind) - u(n+1)*sign(u(ind));
db = 2*phi'*(r - phi*x);
dz(1:n) = db - u(n+1) * (u(1:n)-x);         % Variable neurones (n)
dz(n+1) = norm(x, 1) - l1bound;             % Lagrange neurone  (1)
dz = dz';