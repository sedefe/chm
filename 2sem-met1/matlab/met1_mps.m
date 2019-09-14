function [X, Y] = met1_mps(A, b, x0, eps)
%MET1_MPS	this method should numerically find min(y),
%     where y = 1/2*x'*A*x + b'*x
%     :param A: matrix NxN
%     :param b: matrix Nx1
%     :param x0: matrix Nx1
%     :param eps: accuracy (see test_met1())
%     :return: x and y history across all iterations

% this is dummy code, you should implement your own
[X, Y] = met1_mngs(A, b, x0, eps/2);
