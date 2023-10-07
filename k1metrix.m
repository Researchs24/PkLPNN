function k1x = k1metrix(xp, kp)
% k1metrix computes ||xp||_{kp,1} according to the expression
% ||xp||_{kp,1} = \sum_{i=1}^n |xp_i|_k
% where |xp_i|_k = \frac{ln(2(1 + cosh(xp kp)))}{kp}.
%   k1x = k1metrix(xp, kp) is the ||xp||_{1,kp}. xp is a vector and kp is
%   the k' value.
% v1.6 rv2.0
    syms sk
    f = @(sx, sk) log(2 + 2*cosh(sk * sx))/sk;

    k1x = double(sum(f(xp, vpa(kp))));
end