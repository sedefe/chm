close all
clear all
clc
format long

%%
student = 0; % here is your number

N = student;
A = [
    4,  1,      1
    1,  6+.2*N, -1
    1,  -1,     8+.2*N
    ];
b = [1, -2, 3]';
x0 = [0, 0, 0]';
eps = 1e-6;

methods = {'met1_mngs', 'met1_mps'};

styles = {'mo-', 'b.:'};

hold on
for i = 1:numel(methods)
    method = methods{i};
    disp(['running ', method, ' test for student #', num2str(student)])
    [X, Y] = feval(method, A, b, x0, eps);
    
    x1 = linsolve(A, -b);
    y1 = 1/2*x1'*A*x1 + b'*x1;
    
    assert(norm(x1 - X(:, end)) < 1e-3)
    assert(norm(y1 - Y(:, end)) < eps)
    plot(-log10(Y - y1), styles{i})
end
xlabel('номер итерации')
ylabel('точность')
legend show
