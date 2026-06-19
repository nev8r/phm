MAIN=main

all:
	latexmk -xelatex -interaction=nonstopmode -halt-on-error $(MAIN).tex

clean:
	latexmk -C
	rm -f *.bbl *.blg *.aux *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.xdv

.PHONY: all clean
