function normV = computeNorm(nMethod, x, k)
    switch(nMethod)
        case{1, 2}
            normV = k1metrix(x, k);
        case 3
            normV = k1metrix2(x, k);
        otherwise
            normV = norm(x, 1);
    end
end
%% 
% %% k1metrix2

function mt = k1metrix2(x, k)
    f = @(px, pk) sqrt(px.^2 + 1/pk^2);
    
    mt = sum(f(x, k));
end