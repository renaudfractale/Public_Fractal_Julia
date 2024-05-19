/usr/local/cuda/bin/nvcc -c src/main.cu -o bin/main.o -I/usr/local/cuda/lib64 -I/usr/local/cuda/extras/CUPTI/lib64
g++ -c -I/usr/local/cuda/include  src/main.cpp -o bin/main_cpp.o 
g++ bin/main.o bin/main_cpp.o -o main -lcudart -L/usr/local/cuda/lib64 -L/usr/local/cuda/extras/CUPTI/lib64