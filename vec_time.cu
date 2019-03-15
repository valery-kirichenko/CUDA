#include <iostream>
#include <random>
#include <iomanip>

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

    cudaEvent_t start, stop;
    float worktime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int blocks_opts[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};

    for (int i = 0; i < 10; ++i) {
        cudaEventRecord(start, 0);
        kernel<<<blocks_opts[i], size / blocks_opts[i]>>>(dev_c, dev_a, dev_b, size);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&worktime, start, stop);
        std::cout << blocks_opts[i] << ", " << size / blocks_opts[i] << ":\t" << std::fixed << std::setprecision(16) << worktime << std::endl;
    }

    cudaDeviceSynchronize();

    cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    print_short(c, size, 5);
    
    return 0;
}