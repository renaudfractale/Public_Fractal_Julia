nvcc -c src/main.cu -o bin/main.o
g++ -c src/main.cpp -o bin/main_cpp.o
g++ bin/main.o bin/main_cpp.o -o main -lcudart -L/usr/local/cuda/lib64