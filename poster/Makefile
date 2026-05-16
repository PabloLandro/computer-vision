TEX = poster
PDF = $(TEX).pdf

all: $(PDF)

$(PDF): $(TEX).tex
	pdflatex -interaction=nonstopmode $(TEX).tex
	pdflatex -interaction=nonstopmode $(TEX).tex

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb

distclean: clean
	rm -f $(PDF)

.PHONY: all clean distclean
