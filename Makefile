objects = component.o layers.o main.o
CC = g++
obj = main
CFLAGS = -fopenmp

main:$(objects)
	$(CC) -o $(obj) $(objects) $(CFLAGS)

component.o : component.h
	$(CC) -c component.cpp $(CFLAGS)
layers.o : component.h layers.h
	$(CC) -c component.cpp layers.cpp $(CFLAGS)
main.o : component.h layers.h
	$(CC) -c component.cpp layers.cpp main.cpp $(CFLAGS)

.PHONY:clean
clean :
	-rm main $(objects)