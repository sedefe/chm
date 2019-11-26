function P = met2_interpol(X, Y)
%MET2_INTERPOL	this method should find polynomial interpolation 
%     :param X: X-values (1xN)
%     :param Y: Y-values (1xN)
%     :return: coefficients of N-1-degree polynome P (1xN)

% this is dummy code, you should implement your own
P = polyfit(X, Y, numel(X) - 1);
