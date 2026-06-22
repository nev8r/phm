MAIN=main

all:
	xelatex $(MAIN).tex
	bibtex $(MAIN) || true
	xelatex $(MAIN).tex
	xelatex $(MAIN).tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot
