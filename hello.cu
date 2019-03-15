#include "stdio.h"

__global__ void mykernel() {

}

int main() {
    mykernel<<<1,1>>>();
    printf("Hello world\n");
    return 0;
}