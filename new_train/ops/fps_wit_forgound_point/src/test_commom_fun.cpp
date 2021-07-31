//
// Created by liang on 2020/9/1.
//
#include "math.h"
#include "iostream"
#include "stdio.h"
int main(){
    using namespace std;
    int pow2= std::log(10000)/log(2);
    cout<<std::log(10000)/log(2)<<endl;
    auto a = 1<<pow2;
    cout<<a<<endl;
}
