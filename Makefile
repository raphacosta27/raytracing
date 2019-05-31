PROGS= main \
       par1 \
	   par2 \
	   par3 \
	   par4 \
	   simd

all : $(PROGS)

raw : main.cpp
	g++-8 main.cpp -o main

par1: 
	g++-8 par1.cpp -o par1 -fopenmp

par2:
	g++-8 par2.cpp -o par2 -fopenmp

par3:
	g++-8 par3.cpp -o par3 -fopenmp

par4:
	g++-8 par2.cpp -o par4 -fopenmp

simd:
	g++-8 par3.cpp -o simd -fopenmp -mavx2

clean:
	rm -f $(PROGS)
