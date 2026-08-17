# thesis-tex/

LaTeX source of the thesis *Bio-Inspired Foveation in DeepGaze III*, Imer Itez, Goethe-Universität Frankfurt am Main. The built PDF is `main.pdf`.

```
thesis-tex/
├── main.tex                ← document class, preamble, \include of chapters
├── main.pdf                ← the built thesis
├── macros.tex              ← project-specific \newcommands
├── bibliography.bib        ← references
├── vocabulary.tex          ← List of Vocabulary (front matter)
├── ch01_introduction.tex
├── ch02_background.tex
├── ch03_methodology.tex
├── ch04_results.tex
├── ch05_discussion.tex
├── ch06_conclusion.tex
├── appendix_implementation.tex
└── figures-external/       ← figures reproduced from published papers (README inside)
```

Figures come from `../results/` (this project's own output) and `figures-external/`, through
`\graphicspath{{../results/}{figures-external/}}`, so

```latex
\includegraphics[width=\linewidth]{foveation_mit1003/figs/gp_pyramid.png}
```

picks up `../results/foveation_mit1003/figs/gp_pyramid.png`. When a PNG is regenerated, the next build picks it up.

## Build

Needs `latexmk`, `biber` and the packages `biblatex`, `csquotes`, `microtype`, `subcaption`, `koma-script`, `dsfont` (on BasicTeX: `sudo tlmgr install latexmk biber biblatex csquotes microtype subcaption koma-script dsfont`).

```bash
cd thesis-tex/
latexmk -pdf -interaction=nonstopmode main.tex            # full build
latexmk -pdf -pvc main.tex                                # watch mode: rebuild on save
latexmk -c                                                # remove build artefacts, keep main.pdf
```

## Conventions

- One sentence per line in `.tex` source.
- Section labels `\label{sec:per-image-diagnostic}`, figure labels `\label{fig:gp-pyramid}`; refer to figures with `\figref{...}` from `macros.tex`.
- Citations `\cite{key2022}`, author-year; bib keys are `firstauthorYEARshorttitle`.
