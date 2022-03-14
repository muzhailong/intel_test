objects = component.o layers.o main.o
CXX = g++
obj = main
CFLAGS = -fopenmp

main:$(objects)
	$(CXX) -o $(obj) $(objects) $(CFLAGS)

component.o : component.h
	$(CXX) -c component.cpp $(CFLAGS)
layers.o : component.h layers.h
	$(CXX) -c component.cpp layers.cpp $(CFLAGS)
main.o : component.h layers.h
	$(CXX) -c component.cpp layers.cpp main.cpp $(CFLAGS)

.PHONY:clean
clean :
	-rm main $(objects)