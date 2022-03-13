#ifndef LAYERS_H__
#define LAYERS_H__
#include<iostream>
#include<memory>
#include"component.h"
#include<omp.h>
#include <string>
using namespace std;
#define dtn(n,min_n,tn) \


class Module{
public:
    virtual shared_ptr<Tensor>forward(shared_ptr<Tensor>)=0;
    virtual shared_ptr<Tensor>backward(shared_ptr<Tensor>)=0;
};

class Layer:public Module{
public:
    string name;
    Layer(string name="None");
};

class Relu:public Layer{
public:
    Relu(string name="Relu");
    shared_ptr<Tensor>forward(shared_ptr<Tensor>);
    shared_ptr<Tensor>backward(shared_ptr<Tensor>){};
};

class BatchNorm1D:public Layer{};

class BatchNorm2D:public Layer{
public:
    BatchNorm2D(Dim input_dims,string name="BatchNorm2D",float eps=1e-6);
    shared_ptr<Tensor>forward(shared_ptr<Tensor>);
    shared_ptr<Tensor>backward(shared_ptr<Tensor>){};
protected:
    Dim input_dims;
    shared_ptr<Tensor>global_mean;
    shared_ptr<Tensor>global_std;
    shared_ptr<Tensor>scale;
    shared_ptr<Tensor>shift;
    float eps;
};

class BatchNorm3D:public Layer{};

class Conv1D:public Layer{};
class Conv2D:public Layer{
public:
    Conv2D(size_t units,Dim kernel_size,size_t stride =1,size_t padding=0,string name="Conv2D");
    shared_ptr<Tensor>forward(shared_ptr<Tensor>);
    shared_ptr<Tensor>backward(shared_ptr<Tensor>){};
protected:
    size_t units;
    Dim kernel_size;
    vector<shared_ptr<Tensor>>kernels;
    size_t padding;
    size_t stride;
};
class Conv3D:public Layer{};
#endif