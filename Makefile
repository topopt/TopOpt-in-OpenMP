CFLAGS ?= -Wall -g -fno-omit-frame-pointer -fopenmp -O3 -march=native
LIBS ?= -lm -lopenblas -lcholmod
CC ?= gcc
CXX ?= g++

OBJ = stencil_methods.o multigrid_solver.o grid_utilities.o stencil_optimization.o local_matrix.o coarse_assembly.o coarse_solver.o write_vtk.o

all: top3d

top3d: top3d.c $(OBJ)
	$(CC) -std=c11 $(CFLAGS) -o $@ $^ $(LIBS)

benchmark: benchmark.cpp $(OBJ)
	$(CXX) -std=c++11 $(CFLAGS) -o $@ $^ $(LIBS) -lbenchmark -lpthread

test_stencil_methods: test_stencil_methods.c $(OBJ)
	$(CC) -std=c11 $(CFLAGS) -o $@ $^ $(LIBS)

%.o: %.c
	$(CC) -std=c11 $(CFLAGS) -o $@ -c $<

clean:
	-rm -f benchmark test_stencil_methods core *.core *.o 
