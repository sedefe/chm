function [Y1, a, P] = met3_approximate(X0, Y0, X1, approx_type, dim)
%MET3_APPROXIMATE	this method should perform approximation on [-1; 1] interval
%     :param X0: X-values (1 x N0) 
%     :param Y0: Y-values (1 x N0)
%     :param X1: approximation points (1 x N1)
%     :param approx_type: 
%           0 - algebraic polynomes (1, x, x^2, ...)
%           1 - legendre polynomes
%           2 - harmonic
%     :param dim: dimension
%     :return Y1: approximated Y-values (1 x N1)
%     :return a: vector (1 x dim) of approximation coefficients
%     :return P: (for approx_type 0 and 1) coefficients of approximation polynome P (1 x dim)

% this is dummy code, you should implement your own


%% algebraic polynomes
if approx_type == 0
    Y1 = met3_func(X1) + 1e-10 * randn(size(X1));
    a = [1 zeros(1, dim-1)];
    P = [1 zeros(1, dim-1)];
    return
end


%% legendre polynomes
if approx_type == 1
    Y1 = met3_func(X1) + 1e-10 * randn(size(X1));
    a = [1 zeros(1, dim-1)];
    P = [1 zeros(1, dim-1)];
    return
end


%% harmonic
if approx_type == 2
    N = numel(X0);
    dim = floor(dim/2);
    Q = zeros(N,2*dim+1);
    Q(:,1) = mean(Y0);
    for j=2:dim+1
        Q(:,j) = cos(pi * (j-1) * X0);
        Q(:,dim+j) = sin(pi * (j-1) * X0);
    end
    
    H = Q' * Q;
    b = Q' * Y0;
    a = H \ b;

    Y1 = X1*0 + a(1);
    for j=2:dim+1
        Y1 = Y1 + cos(pi * (j-1) * X1) * a(j);
        Y1 = Y1 + sin(pi * (j-1) * X1) * a(dim+j);
    end
    
    return
end

%%
error(['approximation of type ' num2str(approx_type) ' not supported yet'])
