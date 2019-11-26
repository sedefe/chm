function [X, Y] = met1_mngs(A, b, x0, eps)
%MET1_MNGS	this method should numerically find min(y),
%     where y = 1/2*x'*A*x + b'*x
%     :param A: matrix NxN
%     :param b: matrix Nx1
%     :param x0: matrix Nx1
%     :param eps: accuracy (see test_met1())
%     :return: x and y history across all iterations

% this is dummy code, you should implement your own
x1 = linsolve(A, -b);
P = linspace(0, 1-eps, 10);
X = [];
Y = [];
for p = P
    x = x1*p + x0*(1-p);
    X = [X x];
    Y = [Y 1/2*x'*A*x + b'*x];
end
