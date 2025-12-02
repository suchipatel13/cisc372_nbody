FLAGS = -DDEBUG
LIBS = -lm

nbody: nbody.o compute_cuda.o
	nvcc $(FLAGS) $^ -o $@ $(LIBS)

nbody.o: nbody.c planets.h config.h vector.h
	gcc $(FLAGS) -c nbody.c

compute_cuda.o: compute_cuda.cu config.h vector.h
	nvcc $(FLAGS) -c compute_cuda.cu

clean:
	rm -f *.o nbody

