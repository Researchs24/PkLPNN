function odeSol = odeSolver(data, simParam, setting, nSolver)
%  ODESOLVER Summary of this function goes here v1.5 rv2.0
options=odeset('RelTol', 1.2e-4, 'AbsTol', 1.e-4,'Events', @eventFnc);%,...
%'OutputSel', find(x), 'OutputFcn', @odeplot);
z0 = [data.x0; simParam.lambda0];
global nn;
nn = 0;
tic;
switch(nSolver)
    case 1
        % pk-LPNN v1
        [odeSol.tz,odeSol.xss]=ode113(@(tt,z)pklpnn1(tt,z,data.phi,data.r,...
            setting.eta, setting.k),...
        simParam.tspan,z0,options);
        odeSol.timeode = toc();
        zm = size(odeSol.xss, 1);
        lambdas = odeSol.xss(:,end);
        odeSol.xss = splitapply(@(xx) T(xx(1:end-1), xx(end)),...
            odeSol.xss(:, 1:end)', 1:zm)';
        odeSol.xss = [odeSol.xss lambdas];
    case 2
        % pk-LPNN v2
        [odeSol.tz,odeSol.xss]=ode113(@(tt,z)pklpnn2(tt,z,data.phi,data.r,...
            setting.eta, setting.k),...
        simParam.tspan,z0,options);
        odeSol.timeode = toc();
        zm = size(odeSol.xss, 1);
        lambdas = odeSol.xss(:,end);
        odeSol.xss = splitapply(@(xx) T(xx(1:end-1), xx(end)),...
            odeSol.xss(:, 1:end)', 1:zm)';
        odeSol.xss = [odeSol.xss lambdas];
    case 3
        % pk-LPNN new
        [odeSol.tz,odeSol.xss]=ode45(@(tt,z)pklpnnNew(tt,z,data.phi,data.r,...
            setting.eta, setting.k),...
        simParam.tspan,z0,options);
        odeSol.timeode = toc();
        zm = size(odeSol.xss, 1);
        lambdas = odeSol.xss(:,end);
        odeSol.xsst = splitapply(@(xx) T1(xx(1:end-1), .5, simParam.alpha),...
            odeSol.xss(:, 1:end)', 1:zm)';
        odeSol.xsst = [odeSol.xsst lambdas];
        odeSol.xss = splitapply(@(xx) T(xx(1:end-1), xx(end)),...
            odeSol.xss(:, 1:end)', 1:zm)';
        odeSol.xss = [odeSol.xss lambdas];
    case 4
        % LASSO-LPNN
        [odeSol.tz,odeSol.xss]=ode113(@(tt,z)lassolpnn(tt,z,data.phi,data.r,setting.eta),...
        simParam.tspan,z0,options);
        odeSol.timeode = toc();
        zm = size(odeSol.xss, 1);
        lambdas = odeSol.xss(:,end);
        odeSol.xss = splitapply(@(xx) T(xx(1:end-1), xx(end)),...
            odeSol.xss(:, 1:end)', 1:zm)';
        odeSol.xss = [odeSol.xss lambdas];
    case 5
        % LassoConstrained
        odeSol.xss = LassoConstrained(data.phi, data.r, setting.eta)';
        odeSol.timeode = toc();
        odeSol.tz = 1:length(odeSol.xss);
    otherwise
        % LassoActiveSet
        odeSol.xss = LassoActiveSet(data.phi, data.r, setting.eta)';
        odeSol.timeode = toc();
        odeSol.tz = 1:length(odeSol.xss);
end
% Compute errors with mse
% TODO: add 
switch nSolver
    case {1, 2, 3}
        zm = size(odeSol.xss, 1);
        odeSol.odeerr = splitapply(@(xx) immse(data.x, xx(1:end-1)),...
        odeSol.xss', 1:zm)';
        odeSol.odeRel01loss = single(splitapply(@(xx) rel01loss(data.x, xx(1:end-1)),...
        odeSol.xsst', 1:zm)');
    otherwise   % Only one error
        odeSol.odeerr = immse(data.x', odeSol.xss);
end
odeSol.steps = nn;
end
%%
% Soft thresholding
function x = T(u, lambda)
    ind = abs(u) > lambda;
    x = zeros(length(u),1);
    x(ind) = u(ind) - lambda*sign(u(ind));
end
% Numerical and soft thresholding
function x = T1(u, lambda, alpha)
    ind = abs(u) > lambda;
    x = zeros(length(u),1);
    x(ind) = u(ind) - lambda*sign(u(ind));
    x(ind) = alpha * sign(x(ind));
end
% Relative 0-1 loss
function err = rel01loss(trueSignal, predictedSignal)
    err = sum(trueSignal ~= predictedSignal) / length(trueSignal);
end