function odeSol = odeSolver(data, simParam, setting, nSolver)
%odeSolver Summary of this function goes here  v1.5 rv2.0

options=odeset('RelTol', 1.e-4, 'AbsTol', 1.e-4,'Events', @eventFnc);%,...
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
        % LASSO-LPNN
        [odeSol.tz,odeSol.xss]=ode113(@(tt,z)lassolpnn(tt,z,data.phi,data.r,setting.eta),...
        simParam.tspan,z0,options);
        odeSol.timeode = toc();
        zm = size(odeSol.xss, 1);
        lambdas = odeSol.xss(:,end);
        odeSol.xss = splitapply(@(xx) T(xx(1:end-1), xx(end)),...
            odeSol.xss(:, 1:end)', 1:zm)';
        odeSol.xss = [odeSol.xss lambdas];
    case 4
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



switch nSolver
    case {1, 2, 3}
        zm = size(odeSol.xss, 1);
        odeSol.odeerr = splitapply(@(xx) immse(data.x, xx(1:end-1)),...
        odeSol.xss', 1:zm)';
    otherwise   % Only one error
        odeSol.odeerr = immse(data.x', odeSol.xss);
end
odeSol.steps = nn;
end

function x = T(u, lambda)
ind = abs(u) > lambda;
x = zeros(length(u),1);
x(ind) = u(ind) - lambda*sign(u(ind));
end
