function dxdt = lorenz63Rhs(~, x, sigma, rho, beta)
%LORENZ63RHS Lorenz-63 right-hand side.
if nargin < 3, sigma = 10; end
if nargin < 4, rho = 28; end
if nargin < 5, beta = 8/3; end

dxdt = [
    sigma * (x(2) - x(1));
    x(1) * (rho - x(3)) - x(2);
    x(1) * x(2) - beta * x(3)
];
end
