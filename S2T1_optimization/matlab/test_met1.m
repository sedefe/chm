close all
clear all
clc
format long

%%
student = 0; % here is your number

for n_dim = [2, 3]
    N = student;
    A = [
        4,  1,      1
        1,  6+.2*N, -1
        1,  -1,     8+.2*N
        ];
    b = [1, -2, 3]';
    x0 = [0, 0, 0]';

    A = A(1:n_dim, 1:n_dim);
    b = b(1:n_dim);
    x0 = x0(1:n_dim);
    
    eps_y = 1e-6;
    eps_x = 1e-3;

    methods = {'met1_mngs', 'met1_mps'};

    styles = {'mo-', 'b.:'};

    figure(n_dim)
    title(['Результаты для размерности ' num2str(n_dim)])
    hold on
    for i = 1:numel(methods)
        method = methods{i};
        disp(['running ', method, ' test for student #', num2str(student)])
        [X, Y] = feval(method, A, b, x0, eps_y);

        x1 = linsolve(A, -b);
        y1 = 1/2*x1'*A*x1 + b'*x1;

        assert(all(x0 == X(:,1)), ...
            'X should start with initial point')
        assert(norm(x1 - X(:, end)) < eps_x, ...
            'last X should be close enough to the optimum')
        assert(norm(y1 - Y(:, end)) < eps_y, ...
            'last Y should be close enough to the optimum')
        plot(-log10(Y - y1), styles{i})
    end
    xlabel('номер итерации')
    ylabel('точность')
    legend show
end
