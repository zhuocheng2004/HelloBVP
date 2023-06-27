# 
# Author: Zhuo Cheng
#


.PHONY: _all
_all: all

MAIN		?= info

DVIPDF		?= dvipdf
LATEX		?= latex
PDFLATEX	?= pdflatex

interobjexts	:= .aux .log .toc .out
interobjs	:= $(addprefix $(MAIN), $(interobjexts))
target		:= $(addsuffix .pdf, $(MAIN))


png_names	:= demo_sigma demo_1_1 demo_1_2 demo_2_1 demo_2_2
pngs		:= $(addsuffix .png, $(png_names))

demo_py_files	:= $(shell find hellobvp/demos/ -name '*.py')

other_build	:= build/ 

export MAIN

# Definitons end here

.PHONY: all
all: $(MAIN).pdf


$(MAIN).pdf: $(MAIN).tex $(pngs)
	$(PDFLATEX) $^
	$(PDFLATEX) $^

$(pngs): run_demos.sh $(demo_py_files)
	@./run_demos.sh nogui

.PHONY: clean
clean:
	rm -f $(interobjs) $(images-dep) $(target) $(pngs)
	rm -rf $(other_build)

