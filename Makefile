PROGS= main \
      mainRaw 

all : $(PROGS)

main : main.cu
	nvcc main.cu -o main

mainRaw : raw/main.cpp
	g++-9 raw/main.cpp -o raw/mainRaw

clean:
	rm -f $(PROGS)