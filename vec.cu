#include <iostream>
#include <random>

__global__ void kernel(int* c, int* a, int* b, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        c[i] = a[i] + b[i];
    }
}


void print_short(int* a, int size, int border) {
    for (int i = 0; i < border; ++i) {
        std::cout << a[i] << "\t";
    }
    std::cout << "..\t";
    for (int i = size - border; i < size; ++i) {
        std::cout << a[i] << "\t";
    }
    std::cout << std::endl;
}


int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 10);

    int size = 512, a[size], b[size], c[size];
    for (int i = 0; i < size; ++i) {
        a[i] = dis(gen);
        b[i] = dis(gen);
    }
    print_short(a, size, 5);
    print_short(b, size, 5);

    int *dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr;
    cudaMalloc((void**)&dev_a, size * sizeof(int));
    cudaMalloc((void**)&dev_b, size * sizeof(int));
    cudaMalloc((void**)&dev_c, size * sizeof(int));

    cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    int blocks = 16, threads = size / blocks;
    kernel<<<blocks, threads>>>(dev_c, dev_a, dev_b, size);

    cudaDeviceSynchronize();

    cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    print_short(c, size, 5);
    
    return 0;
}