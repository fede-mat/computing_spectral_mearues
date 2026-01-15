clear all; close all;

%% Matrix and vector definition
A = [1 2; 2 1];
R =norm(A,'fro');
f = [0; 1];

%% Exact eigenvalues (reference)
eigvals = sort(eig(A));

%% Regularized spectral measure (Stone formula)
mu_f_eps = @(A,f,x,eps) ...
    imag( f' * ((A - (x + 1i*eps)*eye(size(A))) \ f) ) / pi;

%% Spatial grid
N = 750;
x = linspace(-R*1.5, R*1.5, N);

%% Evaluate spectral density for fixed epsilon
eps01 = 0.1;
eps001 = 0.01;
mu_vals_01 = zeros(1, N);
mu_vals_001 = zeros(1,N);
for k = 1:N
    mu_vals_01(k) = mu_f_eps(A, f, x(k), eps01);
    mu_vals_001(k) = mu_f_eps(A, f, x(k), eps001);
end

%% Plot
figure;
plot(x, mu_vals_01, 'LineWidth', 1.5); hold on;
plot(x, mu_vals_001, 'LineWidth',1.5);
ylabel('$\mu_{f,\varepsilon}(x)$', 'Interpreter', 'latex');
title('Regularized spectral measure', 'Interpreter', 'latex');
grid on;

%% Peak detection (eigenvalue estimation)
[pks, locs] = findpeaks(mu_vals_001, x, 'MinPeakProminence', 0.1);

disp('Estimated eigenvalues from peaks:');
disp(locs);

disp('Exact eigenvalues:');
disp(eigvals);

%% Dependence on epsilon
M = 750;
eps_values = linspace(1e-3, eps01, M);
mu_surface = zeros(N, M);

for i = 1:N
    for j = 1:M
        mu_surface(i,j) = mu_f_eps(A, f, x(i), eps_values(j));
    end
end

%% Surface plot
figure;
surf(eps_values, x, mu_surface, 'EdgeColor', 'none');
xlabel('$\varepsilon$', 'Interpreter', 'latex');
ylabel('$x$', 'Interpreter', 'latex');
zlabel('$\mu_{f,\varepsilon}(x)$', 'Interpreter', 'latex');
title('Convergence of the spectral measure', 'Interpreter', 'latex');
view(45,30);

%%
%% Convergence order analysis (log-log scale)

% Point where we test convergence (choose an eigenvalue)
x0 = eigvals(2);   % e.g. largest eigenvalue

% Exact spectral weight at x0
% (projection of f onto the eigenspace)
[V,D] = eig(A);
[~,idx] = min(abs(diag(D) - x0));
v = V(:,idx);
mu_exact = abs(f' * v)^2;

% Epsilon values (log-spaced)
eps_conv = logspace(-4, -1, 30);

% Error computation
errors = zeros(size(eps_conv));
for k = 1:length(eps_conv)
    mu_eps = mu_f_eps(A, f, x0, eps_conv(k));
    errors(k) = abs(mu_eps - mu_exact);
end

% Log-log plot
figure;
loglog(eps_conv, errors, 'o-', 'LineWidth', 1.5);
grid on;
xlabel('$\varepsilon$', 'Interpreter', 'latex');
ylabel('Error', 'Interpreter', 'latex');
title('Convergence of $\mu_{f,\varepsilon}(x_0)$ (log--log scale)', ...
      'Interpreter', 'latex');

% Estimate convergence order
p = polyfit(log(eps_conv), log(errors), 1);
order = p(1);

disp('Estimated order of convergence:');
disp(order);

