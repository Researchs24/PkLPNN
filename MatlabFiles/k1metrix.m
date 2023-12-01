function k1x = k1metrix(xp, kp)
%  K1METRIX computes $\|x\|_{k,1}$ according to the expression
% 
% $\|x\|_{k,1}=\sum_{i=1}^n|x_i|_k$ where $|x_i|_k= \frac{\ln(2(1 + \cosh(x 
% \cdot k)))}{k}$.
% 
% k1x = k1metrix(xp, kp) is the ||xp||_{1,kp}. xp is a vector and kp is the 
% k' value. v1.6 rv2.0
    syms sk
    f = @(sx, sk) log(2 + 2*cosh(sk * sx))/sk;
    k1x = double(sum(f(xp, vpa(kp))));
end