function [value,isterminal,direction] = eventFnc(t,y)
%  EVENTFNC v1.5 rv2.0 eventFnc is used to configure the ODE behavior
% 
% eventFnc is used within the ODE configuration
    % TODO: Plot the t's values
    global nn
    if isempty(nn)
        nn = 0;
    end
    nn = nn+1;
    
    %[mm, nnz] = size(y);
    %fprintf('inside eventFnc ||y(%d, %d)||1,k = ?\n', mm, nnz);%,...
        %k1metrix(kparam,y(1:end-1),0));
    if(mod(nn, 50000) == 0)
        fprintf('nn = %5d\tt = %10.6f\n', nn, t);
        %memory;
    end
    value      = true;%(nn <= 10);%(nn <= 100000);%(nn <= 10);%true;%true;%true; %10% 1 == 1;%;%10000%4000; true;%
    isterminal = 1;   % Stop the integration
    direction  = 0;
end

