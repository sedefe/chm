close all
clear all
clc

%%
N = 5;
M = N*10;

A = -1; % change to what you prefer
B = 3; % change to what you prefer

X_eq = linspace(A, B, N+1);
X_cheb = 1/2 * ((B-A) * cos(pi * (2*(0:N) + 1) / (2*(N + 1))) + (B+A));
X_dense = sort([linspace(A, B, M) X_eq X_cheb]);
Y_dense = met2_func(X_dense);

figure(1), hold on
plot(X_dense, Y_dense, 'k-', ...
    'DisplayName', 'actual')
styles = {'b.:', 'm.:'};

Xs = {X_eq, X_cheb};
labels = {'interp-eq', 'interp-cheb'};
for i = 1:numel(Xs)
    X = Xs{i};    
    Y0 = met2_func(X);  % here's your function (change met2_func.m)
    
    P = met2_interpol(X, Y0);  % here's your interpolation (change met2_interpol.m)
    
    assert(numel(P) == N+1, ...
        ['polynome length should be ' num2str(N+1)])
    
    Y2 = polyval(P, X_dense);
    
    figure(1), hold on
    plot(X_dense, Y2, styles{i}, ...
        'DisplayName', labels{i})
    figure(2), hold on
    plot(X_dense, log10(abs(Y_dense - Y2)), styles{i}, ...
        'DisplayName', labels{i})
end

%% spline
Y_spline = spline(X_eq, met2_func(X_eq), X_dense);
figure(1), hold on
plot(X_dense, Y_spline, 'c-', 'DisplayName', 'spline');
figure(2), hold on
plot(X_dense, log10(abs(Y_dense - Y_spline)), 'c-', ...
    'DisplayName', 'spline')

%%
figure(1), title('Y(X)'), 
legend show
figure(2), title('log error')
legend show
