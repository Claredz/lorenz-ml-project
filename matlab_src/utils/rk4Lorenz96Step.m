function xNext = rk4Lorenz96Step(x, dt, forcing)
%RK4LORENZ96STEP One RK4 step for Lorenz-96 in physical coordinates.
if nargin < 3
    forcing = 8.0;
end
rowInput = isrow(x);
x = x(:);
k1 = lorenz96Rhs(0, x, forcing);
k2 = lorenz96Rhs(0, x + 0.5 * dt * k1, forcing);
k3 = lorenz96Rhs(0, x + 0.5 * dt * k2, forcing);
k4 = lorenz96Rhs(0, x + dt * k3, forcing);
xNext = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4);
if rowInput
    xNext = xNext.';
end
end
