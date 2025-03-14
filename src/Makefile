include Make_linux.inc
#include Make_msys2.inc
#include Make_osx.inc

CXXFLAGS = -std=c++17
ifdef DEBUG
CXXFLAGS += -g -O0 -Wall -fbounds-check -pedantic -D_GLIBCXX_DEBUG
CXXFLAGS2 = CXXFLAGS
else
CXXFLAGS2 = ${CXXFLAGS} -O2 -march=native -Wall
CXXFLAGS += -O3 -march=native -Wall
endif

ALL= simulation.exe 

default:	help

all: $(ALL)

clean:
	@rm -fr *.o *.exe *~

.cpp.o:
	$(MPICXX) $(CXXFLAGS2) -c $^ -o $@	

simulation.exe : display.o display.hpp model.o model.hpp simulation.o
	$(MPICXX) $(CXXFLAGS2) $^ -o $@ $(LIB)	

help:
	@echo "Available targets : "
	@echo "    all            : compile all executables"
	@echo "Add DEBUG=yes to compile in debug"
	@echo "Configuration :"
	@echo "    CXX      :    $(CXX)"
	@echo "    CXXFLAGS :    $(CXXFLAGS)"

%.html: %.md
	pandoc -s --toc $< --css=./github-pandoc.css --metadata pagetitle="OS202 - TD1" -o $@
