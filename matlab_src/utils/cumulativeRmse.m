function cumulative = cumulativeRmse(stepErrors)
%CUMULATIVERMSE Compute sqrt(cumsum(step_error^2) / step_index).
stepErrors = stepErrors(:);
idx = (1:numel(stepErrors))';
cumulative = sqrt(cumsum(stepErrors .^ 2) ./ idx);
end
