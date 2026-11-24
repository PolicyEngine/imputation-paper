% ijmarticle.cls -- the article class with APA citations loaded first.
%
% apacite (APA citations and reference list) has to be loaded before natbib.
% With natbib already in memory, apacite 6.03 skips its natbib-emulation code
% wrongly and stops with "Undefined control sequence \st@rtbibchapter". The
% Pandoc/Quarto template loads natbib in the preamble, before any header
% include, so the only place early enough is the class. With the natbibapa
% option apacite loads natbib itself and leaves the citation commands
% (\citep, \citet, ...) to it.
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ijmarticle}[2026/11/14 article with apacite loaded before natbib]
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass{article}
\RequirePackage[natbibapa]{apacite}
\endinput
