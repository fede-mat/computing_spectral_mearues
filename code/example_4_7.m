% =========================================================
% Discretization of an Operator and Spectral Analysis
% =========================================================

clear all; clc;

%% ---------------------------------------------------------
% 1. Spatial grid and quadrature weights
% ----------------------------------------------------------

N = 200;                              % Number of grid points
x = linspace(-1, 1, N)';              % Spatial grid
h = x(2) - x(1);                      % Grid spacing

% Trapezoidal quadrature weights
w = h * ones(N,1);
w(1)   = h/2;
w(end) = h/2;

%% ---------------------------------------------------------
% 2. Construction of the discrete operator
% ----------------------------------------------------------

% Multiplication operator: (Mx u)(x) = x u(x)
Mx = diag(x);

% Integral operator with Gaussian kernel
K = zeros(N,N);
for i = 1:N
    for j = 1:N
        K(i,j) = exp(-(x(i)^2 + x(j)^2)) * w(j);
    end
end

% Full discrete operator
L = Mx + K;

%% ---------------------------------------------------------
% 3. Spectral decomposition
% ----------------------------------------------------------

[V, D] = eig(L);                      % Eigen-decomposition
lambda = diag(D);                     % Eigenvalues

% Sort eigenvalues and eigenvectors
[lambda, idx] = sort(lambda);
V = V(:, idx);

%% ---------------------------------------------------------
% 4. Spectrum visualization
% ----------------------------------------------------------

figure;
scatter(x, lambda, 10, 'filled'); hold on;
plot(x, x, 'r--', 'LineWidth', 1.5);   % Diagonal lambda = x
yline(max(lambda), 'k--', 'isolated eigenvalue');

xlabel('x');
ylabel('\lambda');
title('Discrete spectrum: \lambda versus x');
grid on;

%% ---------------------------------------------------------
% 5. Extraction of the isolated eigenvalue
% ----------------------------------------------------------

[lambda_iso, ind_iso] = max(lambda);
v_iso = V(:, ind_iso);

fprintf('Estimated isolated eigenvalue: %.6f\n', lambda_iso);

%% ---------------------------------------------------------
% 6. Theoretical eigenfunction
% ----------------------------------------------------------

u_theory = exp(-x.^2) ./ (x - lambda_iso);

% Discrete L^2 normalization
v_iso    = v_iso    / sqrt(sum(v_iso.^2    .* w));
u_theory = u_theory / sqrt(sum(u_theory.^2 .* w));

% Enforce consistent sign
if dot(v_iso, u_theory) < 0
    u_theory = -u_theory;
end

%% ---------------------------------------------------------
% 7. Numerical vs theoretical eigenfunction
% ----------------------------------------------------------

figure;
plot(x, v_iso, 'b', 'LineWidth', 2); hold on;
plot(x, u_theory, 'r--', 'LineWidth', 2);

legend('Numerical eigenvector', 'Theoretical eigenfunction');
xlabel('x');
ylabel('Amplitude');
title('Comparison of numerical and theoretical eigenfunctions');
grid on;

%% =========================================================
% Connection with spectral measures (Stone formula)
% =========================================================

A = L;                                % Operator
f = rand(N,1);                        % Random probe vector
f = f / norm(f, 2);

R = norm(A, 'fro');                   % Spectral radius estimate

%% ---------------------------------------------------------
% 8. Regularized spectral measure
% ----------------------------------------------------------

mu_f_eps = @(A,f,x,eps) ...
    imag( f' * ((A - (x + 1i*eps)*eye(size(A))) \ f) ) / pi;

%% ---------------------------------------------------------
% 9. Spectral grid
% ----------------------------------------------------------

Nx = 750;
x_spec = linspace(-1.5*R, 1.5*R, Nx);

%% ---------------------------------------------------------
% 10. Spectral density for fixed epsilon
% ----------------------------------------------------------

eps1 = 0.1;
eps2 = 0.01;

mu_vals_1 = zeros(1, Nx);
mu_vals_2 = zeros(1, Nx);

for k = 1:Nx
    mu_vals_1(k) = mu_f_eps(A, f, x_spec(k), eps1);
    mu_vals_2(k) = mu_f_eps(A, f, x_spec(k), eps2);
end

%% ---------------------------------------------------------
% 11. Spectral density plot
% ----------------------------------------------------------

figure;
plot(x_spec, mu_vals_1, 'LineWidth', 1.5); hold on;
plot(x_spec, mu_vals_2, 'LineWidth', 1.5);

ylabel('$\mu_{f,\varepsilon}(x)$', 'Interpreter', 'latex');
title('Regularized spectral measure', 'Interpreter', 'latex');
grid on;

%% ---------------------------------------------------------
% 12. Eigenvalue estimation via peak detection
% ----------------------------------------------------------

[pks, locs] = findpeaks(mu_vals_2, x_spec,'MinPeakProminence', 0.1);

%% ---------------------------------------------------------
% 12b. Error analysis: peak-based eigenvalue estimation
% ----------------------------------------------------------
% We compare the eigenvalues detected from the peaks of the
% regularized spectral measure with the exact eigenvalues
% obtained from the eigendecomposition.

% For each detected peak, find the closest exact eigenvalue
num_peaks = length(locs);
peak_errors = zeros(num_peaks,1);
matched_eigenvalues = zeros(num_peaks,1);

for k = 1:num_peaks
    [peak_errors(k), idx_min] = ...
        min(abs(lambda - locs(k)));
    matched_eigenvalues(k) = lambda(idx_min);
end

% Display results
T = table(locs(:), matched_eigenvalues, peak_errors, ...
    'VariableNames', {'EstimatedPeak', 'ExactEigenvalue', 'AbsoluteError'});

disp('Error analysis for peak-based eigenvalue estimation:');
disp(T);

% Summary statistics
fprintf('Maximum absolute error: %.4e\n', max(peak_errors));
fprintf('Mean absolute error:    %.4e\n', mean(peak_errors));





%% ---------------------------------------------------------
% 13. Dependence on the regularization parameter epsilon
% ----------------------------------------------------------

M = 150;
eps_values = linspace(1e-3, eps1, M);
mu_surface = zeros(Nx, M);

for i = 1:Nx
    for j = 1:M
        mu_surface(i,j) = mu_f_eps(A, f, x_spec(i), eps_values(j));
    end
end

%% ---------------------------------------------------------
% 14. Convergence of the spectral measure
% ----------------------------------------------------------

figure;
surf(eps_values, x_spec, mu_surface, 'EdgeColor', 'none');

xlabel('$\varepsilon$', 'Interpreter', 'latex');
ylabel('$x$', 'Interpreter', 'latex');
zlabel('$\mu_{f,\varepsilon}(x)$', 'Interpreter', 'latex');
title('Convergence of the regularized spectral measure', ...
      'Interpreter', 'latex');

view(45,30);
