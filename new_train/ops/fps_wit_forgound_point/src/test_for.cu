#include "cuda.h"
#include "stdio.h"
#include "iostream"
#define THREADS_PER_BLOCK 256
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))

void test1(){
    int pts_num=500;//there are 500 points
    int a = DIVUP(pts_num,THREADS_PER_BLOCK);// assign 2 block, because per block has a limit of 256
    dim3 blocks(a, 3, 1);
    dim3 threads(THREADS_PER_BLOCK);

    std::cout<<a<<std::endl;
}

int main(){
    test1();
}