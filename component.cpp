#include "component.h"
using namespace std;
size_t Dim::size(){
    return dims.size();
}
size_t Dim::elements_size(){
    size_t ret=1;
    for(int t:dims){
        ret*=t;
    }
    return ret;
}

vector<size_t>Dim::shape(){
    return dims;
}

Tensor::Tensor(void*buff,DType dtype,Dim dim,bool inplace):dtype(dtype),dim(dim){
    size_t sm = dim.elements_size();
    if(!inplace){
        sm*=F::num_bytes_dtype(dtype);
        data=(void*)new char[sm];
        memcpy(data,buff,sm);
    }else{
        data=buff;
    }
}

Tensor::Tensor(float*buff,Dim dim):Tensor((void*)buff,DType::Float32,dim){
}
Tensor::Tensor(double*buff,Dim dim):Tensor((void*)buff,DType::Float64,dim){
}
Tensor::Tensor(int*buff,Dim dim):Tensor((void*)buff,DType::Int32,dim){
}
Tensor::Tensor(long*buff,Dim dim):Tensor((void*)buff,DType::Int64,dim){
}

Tensor::~Tensor(){
    free(data);
}
shared_ptr<Tensor> Tensor::like_tensor(bool inplace){
    void*new_data=(void*)new char[dim.elements_size()*F::num_bytes_dtype(dtype)];
    if(inplace){
        memcpy((char*)new_data,(char*)data,dim.elements_size()*F::num_bytes_dtype(dtype));
    }
    return make_shared<Tensor>(new_data,dtype,Dim(dim));
}

namespace F{
    size_t num_bytes_dtype(DType dtype){
        size_t ret=-1;
        switch (dtype)
        {
        case DType::Float32:
            ret=4;
            break;
        case DType::Float64:
            ret=8;
            break;
        case DType::Int32:
            ret=4;
            break;
        case DType::Int64:
            ret=8;
            break;
        default:
            throw "dtype is Error!!!";
        }
        return ret;
    }

    shared_ptr<Tensor>zero_tensor(Dim dim,DType dtype){
        size_t num=dim.elements_size();
        switch (dtype)
        {
        case DType::Float32:
            return make_shared<Tensor>((void*)new float[num](),dtype,dim);
        case DType::Float64:
        return make_shared<Tensor>((void*)new double[num](),dtype,dim);
        case DType::Int32:
            return make_shared<Tensor>((void*)new int[num](),dtype,dim);
        case DType::Int64:
        return make_shared<Tensor>((void*)new double[num](),dtype,dim);
        default:
            throw "dtype is Error!!!";
        }
        return make_shared<Tensor>((void*)new float[num](),dtype,dim);
    }

    shared_ptr<Tensor>one_tensor(Dim dim,DType dtype){
        size_t num=dim.elements_size();
        switch (dtype)
        {
        case DType::Float32:
            {
                float*buff=new float[num];
                fill(buff,buff+num,1.0);
                return make_shared<Tensor>((void*)buff,dtype,dim);
            }
        case DType::Float64:
        {
                double*buff=new double[num];
                fill(buff,buff+num,1.0);
                return make_shared<Tensor>((void*)buff,dtype,dim);
            }
        case DType::Int32:
            {
                int*buff=new int[num];
                fill(buff,buff+num,1);
                return make_shared<Tensor>((void*)buff,dtype,dim);
            }
        case DType::Int64:
        {
                long*buff=new long[num];
                fill(buff,buff+num,1);
                return make_shared<Tensor>((void*)buff,dtype,dim);
            }
        default:
            throw "dtype is Error!!!";
        }
        return make_shared<Tensor>((void*)new float[num](),dtype,dim);
    }

    shared_ptr<Tensor>randn_tensor(Dim dim){
        random_device rd;
        // default_random_engine e(time(NULL));
        mt19937 e(rd());
        normal_distribution<float> nd(0,1.0);
        size_t count=dim.elements_size();
        float*data=new float[count];
        for(int i=0;i<count;++i){
            data[i]=nd(e);
        }
        return make_shared<Tensor>((void*)data,DType::Float32,dim);
    }
}