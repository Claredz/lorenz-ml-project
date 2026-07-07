function dxdt = lorenz96Rhs(~, x, forcing)
%LORENZ96RHS Lorenz-96 right-hand side with periodic boundaries.
if nargin < 3
    forcing = 8.0;
end
x = x(:);
N = numel(x);
dxdt = zeros(N, 1);
for i = 1:N
    ip1 = mod(i, N) + 1;
    im1 = mod(i - 2, N) + 1;
    im2 = mod(i - 3, N) + 1;
    dxdt(i) = (x(ip1) - x(im2)) * x(im1) - x(i) + forcing;
end
end
