#ifndef COMPONENT_H__
#define COMPONENT_H__
#include<iostream>
#include<cstdarg>
#include<vector>
#include<string.h>
#include<memory>
#include<random>
#include<ctime>

using namespace std;

enum DType{
    Float32,
    Float64,
    Int32,
    Int64
};

class Dim{
private:
    vector<size_t>dims;
public:
    Dim(){};
    Dim(size_t d0,size_t d1=0,size_t d2=0,size_t d3=0);
    size_t size()const;
    vector<size_t>shape()const;
    size_t elements_size()const;
    bool operator==(const Dim&);
    bool operator!=(const Dim&);
};

class Tensor{
public:
    DType dtype;
    Dim dim;
    Tensor(void*buff,DType dtype,Dim dim,bool inplace=false);
    Tensor(float*buff,Dim dim);
    Tensor(double*buff,Dim dim);
    Tensor(int*buff,Dim dim);
    Tensor(long*buff,Dim dim);
    virtual ~Tensor();
    shared_ptr<Tensor>like_tensor(bool inplace=false);
    void*data;
};

namespace F{
    size_t num_bytes_dtype(DType);
    shared_ptr<Tensor>zero_tensor(Dim dim,DType dtype=DType::Float32);
    shared_ptr<Tensor>one_tensor(Dim dim,DType dtype=DType::Float32);
    shared_ptr<Tensor>randn_tensor(Dim dim);
}

#endif