function run_all_matlab(mode)
%RUN_ALL_MATLAB One-command MATLAB entry point for the Lorenz ML project.
%   run_all_matlab reproduces the MATLAB implementation artifacts used by
%   report.tex: CSV tables in results/ and figures in figures/.
%
%   Usage from a shell:
%       matlab -batch "run_all_matlab"
%
%   Optional:
%       matlab -batch "run_all_matlab('smoke')"

if nargin < 1
    mode = "full";
else
    mode = string(mode);
end

projectRoot = fileparts(mfilename("fullpath"));
addpath(fullfile(projectRoot, "matlab_src"));
addpath(fullfile(projectRoot, "matlab_src", "utils"));

rng(42, "twister");
ensureFolder(fullfile(projectRoot, "results"));
ensureFolder(fullfile(projectRoot, "results", "advanced"));
ensureFolder(fullfile(projectRoot, "figures"));

fprintf("Running MATLAB Lorenz implementation (%s mode).\n", mode);
runLorenz63Experiments(projectRoot, mode);
runLorenz96Experiments(projectRoot, mode);
generateReportTablesMatlab(projectRoot);
fprintf("MATLAB artifacts written to results/ and figures/.\n");
end
