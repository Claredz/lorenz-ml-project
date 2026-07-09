function buildFinalPdf(projectRoot)
%BUILDFINALPDF Compile report body, export the Word-template cover, and merge.
% Requires xelatex and Microsoft Word automation. The generated final PDF is report.pdf.

if nargin < 1
    projectRoot = fileparts(fileparts(mfilename('fullpath')));
end

oldDir = pwd;
cleanup = onCleanup(@() cd(oldDir)); %#ok<NASGU>
cd(projectRoot);

if exist('report.pdf', 'file')
    copyfile('report.pdf', 'report_body_previous.pdf');
end

runCommand('xelatex -interaction=nonstopmode report.tex');
runCommand('xelatex -interaction=nonstopmode report.tex');
movefile('report.pdf', 'report_body.pdf', 'f');

templatePath = fullfile(projectRoot, 'matlab', '2026', '实习论文模板-评分2026 (1).docx');
coverDocxPath = fullfile(projectRoot, 'cover_filled.docx');
coverPdfPath = fullfile(projectRoot, 'cover.pdf');
createCoverFromWordTemplate(templatePath, coverDocxPath, coverPdfPath);

writeMergeTex(projectRoot);
runCommand('xelatex -interaction=nonstopmode merge_report.tex');
movefile('merge_report.pdf', 'report.pdf', 'f');
fprintf('Final PDF written to %s\n', fullfile(projectRoot, 'report.pdf'));
end

function runCommand(commandText)
[status, output] = system(commandText);
if status ~= 0
    error('Command failed: %s\n%s', commandText, output);
end
end

function createCoverFromWordTemplate(templatePath, coverDocxPath, coverPdfPath)
if ~isfile(templatePath)
    error('Word cover template not found: %s', templatePath);
end
if ~ispc
    error('Word cover export requires Microsoft Word automation on Windows.');
end

word = [];
doc = [];
try
    word = actxserver('Word.Application');
    word.Visible = false;
    word.DisplayAlerts = 0;
    doc = word.Documents.Open(char(templatePath), false, false);

    setParagraphText(doc, 7, ['基于 MATLAB 的高维混沌' char(11) '系统长期预测建模']);
    setParagraphText(doc, 9, '学    院     钱伟长学院');
    setParagraphText(doc, 10, '专    业    数学与应用数学');
    setParagraphText(doc, 11, '学号姓名   25120617 钟兴涛');
    setParagraphText(doc, 12, '学号姓名      25120638 唐亦明');
    setParagraphText(doc, 13, '学号姓名   25120636 戴云天');
    setParagraphText(doc, 14, '学号姓名      25120699 任宇航  25120619 黄宇轩');
    setParagraphText(doc, 15, '课    程  MATLAB及应用（强）');
    setParagraphText(doc, 16, ['打印日期  ' char(datetime('today', 'Format', 'yyyy年M月d日'))]);
    formatCoverParagraph(doc, 7, 24, true);
    formatCoverParagraph(doc, 14, 14, false);
    for i = [9, 10, 11, 12, 13, 15, 16]
        formatCoverParagraph(doc, i, [], false);
    end

    doc.SaveAs2(char(coverDocxPath), 16);
    doc.ExportAsFixedFormat(char(coverPdfPath), 17, false, 0, 3, 1, 1);
    doc.Close(false);
    word.Quit();
catch err
    tryCloseWord(doc, word);
    error('Unable to fill and export Word cover template: %s', err.message);
end
end

function setParagraphText(doc, paragraphIndex, text)
range = doc.Paragraphs.Item(paragraphIndex).Range;
range.End = range.End - 1;
range.Text = char(text);
end

function formatCoverParagraph(doc, paragraphIndex, pointSize, centerText)
range = doc.Paragraphs.Item(paragraphIndex).Range;
range.End = range.End - 1;
try
    range.ParagraphFormat.AddSpaceBetweenFarEastAndAlpha = false;
    range.ParagraphFormat.AddSpaceBetweenFarEastAndDigit = false;
catch
end
if ~isempty(pointSize)
    range.Font.Size = pointSize;
end
if centerText
    range.ParagraphFormat.Alignment = 1;
    range.ParagraphFormat.LeftIndent = 0;
    range.ParagraphFormat.RightIndent = 0;
    range.ParagraphFormat.FirstLineIndent = 0;
end
end

function tryCloseWord(doc, word)
try
    if ~isempty(doc)
        doc.Close(false);
    end
catch
end
try
    if ~isempty(word)
        word.Quit();
    end
catch
end
end

function writeMergeTex(projectRoot)
content = [
"\documentclass[a4paper]{article}" newline ...
"\usepackage[paper=a4paper,margin=0pt]{geometry}" newline ...
"\usepackage{pdfpages}" newline ...
"\begin{document}" newline ...
"\includepdf[pages=-]{cover.pdf}" newline ...
"\includepdf[pages=-]{report_body.pdf}" newline ...
"\end{document}" newline];
writeText(fullfile(projectRoot, 'merge_report.tex'), content);
end

function writeText(filePath, content)
fid = fopen(filePath, 'w', 'n', 'UTF-8');
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s', content);
end
