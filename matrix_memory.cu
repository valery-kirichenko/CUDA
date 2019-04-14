#include <iostream>
#include <random>
#include <cmath>

__global__ void kernel(int* c, int* a, int* b, int size) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < size && iy < size) {
        int offset = iy * size + ix;
        c[offset] = a[offset] + b[offset];
    }
}


void print_short(int** a, int size, int border) {
    for (int i = 0; i < border; ++i) {
        for (int j = 0; j < border; ++j) {
            std::cout << a[i][j] << "\t";
        }
        std::cout << "..\t";
        for (int j = size - border; j < size; ++j) {
            std::cout << a[i][j] << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "..\t\n";
    for (int i = size - border; i < size; ++i) {
        for (int j = 0; j < border; ++j) {
            std::cout << a[i][j] << "\t";
        }
        std::cout << "..\t";
        for (int j = size - border; j < size; ++j) {
            std::cout << a[i][j] << "\t";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}


int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 10);

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    unsigned int size;
    std::cout << "Your GPU has " << free_mem << "bytes of free memory; array element takes " << sizeof(int) << std::endl;
    std::cout << "Max matrix size: " << floor(std::sqrt(free_mem / 12)) << std::endl;
    std::cout << "Enter matrix size: ";
    std::cin >> size;

    std::cout << "Starting host memory allocation" << std::endl;
    int **a = new int*[size], **b = new int*[size], **c = new int*[size];
    a[0] = new int[size * size];
    b[0] = new int[size * size];
    c[0] = new int[size * size];

    for (int i = 1; i < size; ++i) {
        a[i] = a[i - 1] + size;
        b[i] = b[i - 1] + size;
        c[i] = c[i - 1] + size;
    }
    std::cout << "Memory allocated; Starting matrices randomization" << std::endl; 
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            a[i][j] = dis(gen);
            b[i][j] = dis(gen);
        }
    }

    std::cout << "===== A: =====\n";
    print_short(a, size, 3);
    std::cout << "===== B: =====\n";
    print_short(b, size, 3);

    int *dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr;
    cudaError_t malloc_result;
    cudaMalloc((void**)&dev_a, size * size * sizeof(int));
    cudaMalloc((void**)&dev_b, size * size * sizeof(int));
    malloc_result = cudaMalloc((void**)&dev_c, size * size * sizeof(int));
    if (malloc_result != cudaSuccess) {
        std::cout << "Can't allocate device memory" << std::endl;
        return 1;
    } else {
        std::cout << "Device memory successfully allocated" << std::endl;;
    }

    cudaMemcpy(dev_a, a[0], size * size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b[0], size * size * sizeof(int), cudaMemcpyHostToDevice);

    int threads_per_block_dimension = 16;
    dim3 blocks(size / threads_per_block_dimension, size / threads_per_block_dimension);
    dim3 threads(threads_per_block_dimension, threads_per_block_dimension);
    kernel<<<blocks, threads>>>(dev_c, dev_a, dev_b, size);

    cudaDeviceSynchronize();

    cudaMemcpy(c[0], dev_c, size * size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    std::cout << "===== C: =====\n";
    print_short(c, size, 3);

    delete[] a[0];
    delete[] b[0];
    delete[] c[0];
    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}