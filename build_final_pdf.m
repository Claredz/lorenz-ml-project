function build_final_pdf()
%BUILD_FINAL_PDF Root-level wrapper for final PDF generation.
projectRoot = fileparts(mfilename("fullpath"));
addpath(fullfile(projectRoot, "matlab_src"));
buildFinalPdf(projectRoot);
end
