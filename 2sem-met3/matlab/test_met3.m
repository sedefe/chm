clear all
close all
clc

%%
N = 5;
x0 = linspace(-1, 1, N)';
dim = 3;

M = 101;
x1 = linspace(-1, 1, M)';

y0 = met3_func(x0);  % here's your function (change met3_func.m)
y1 = met3_func(x1);

%%
% here's your approximation (change met3_approximate.m)
[y_algpoly, ~, P_algpoly] = met3_approximate(x0, y0, x1, 0, dim);
assert(numel(P_algpoly) == dim, ...
        ['polynome length should be ' num2str(dim)])
assert(all(abs(y1 - y_algpoly) < 1), ...
        'algebraic polynome approximation is too bad')

[y_legpoly, ~, P_legpoly] = met3_approximate(x0, y0, x1, 1, dim);
assert(numel(P_legpoly) == dim, ...
        ['polynome length should be ' num2str(dim)])
assert(all(abs(y1 - y_legpoly) < 1), ...
        'legendre polynome approximation is too bad')
% assert(all(abs(P_algpoly - P_legpoly) < 1e-3))

%%
figure(1), hold on, title('log error')
plot(x1, -log10(abs(y1 - y_algpoly)), 'b-p')
plot(x1, -log10(abs(y1 - y_legpoly)), 'g:*')

figure(2), hold on, title('Y(X)')
plot(x1, y1, 'ko')
plot(x1, y_algpoly, 'b-p')
plot(x1, y_legpoly, 'g:*')
