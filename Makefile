main_objects = component.o layers.o main.o
test_objects = component.o layers.o test.o

CXX = g++
CFLAGS = -fopenmp

all:main test

main:$(main_objects)
	$(CXX) -o $@ $(main_objects) $(CFLAGS)

test:$(test_objects)
	$(CXX) -o $@ $(test_objects) $(CFLAGS)

component.o : component.h
	$(CXX) -c component.cpp $(CFLAGS)
layers.o : component.h layers.h
	$(CXX) -c component.cpp layers.cpp $(CFLAGS)
main.o : component.h layers.h
	$(CXX) -c component.cpp layers.cpp main.cpp $(CFLAGS)
test.o : component.h layers.h
	$(CXX) -c component.cpp layers.cpp test.cpp $(CFLAGS)

.PHONY:clean
clean :
	-rm main test *.o