#include"layers.h"
using namespace std;

size_t get_num_threads(size_t n, size_t min_n=8){
#ifdef _OPENMP
    size_t max_tn = n / min_n; 
    size_t ncore = omp_get_num_procs(); 
    // cout<<"ncore:"<<ncore<<endl;
    size_t tn = max_tn > 2*ncore ? 2*ncore : max_tn;
#else
    size_t tn=1;    
#endif
    if (tn<1){ 
        tn=1; 
    }
    return tn;
}

Layer::Layer(string name):name(name){
}
Relu::Relu(string name):Layer(name){}


shared_ptr<Tensor> Relu::forward(shared_ptr<Tensor>input){
    auto ret=input->like_tensor(true);
    size_t tn;
    size_t e_num=(ret->dim).elements_size();
    size_t i=0;
    if (ret->dtype!=DType::Float32){
        throw "ret dtype is not float32!!!";
    }
    float*buff=(float*)ret->data;
#pragma omp parallel for num_threads(get_num_threads(e_num)) shared(buff)
    for(i=0;i<e_num;++i){
        if(buff[i]<0){
           buff[i]=.0;
        }
    }
    return ret;
}

BatchNorm2D::BatchNorm2D(Dim input_dims,string name,float eps)
:Layer(name),input_dims(input_dims),eps(eps){
    auto vec=input_dims.shape();
    if(vec.size()!=2){
        throw "vec shape is Error!!!";
    }
    global_mean= F::zero_tensor(input_dims,DType::Float32);
    global_std= F::zero_tensor(input_dims,DType::Float32);

    scale= F::one_tensor(input_dims,DType::Float32);
    shift= F::randn_tensor(input_dims);
}
shared_ptr<Tensor> BatchNorm2D::forward(shared_ptr<Tensor> input){
    auto ret=input->like_tensor(true);
    if(ret->dtype!=DType::Float32){
        throw "batch norm ret dtype is not float32 Error!!!";
    }
    size_t tn;
    size_t batch=((input->dim).shape())[0];
    size_t row=(input_dims.shape())[0];
    size_t width=(input_dims.shape())[1]; 
    size_t e_num=row*width;
    size_t i=0;
    float*tmp=(float*)ret->data;
    float*scale_tmp=(float*)scale->data;
    float*shift_tmp=(float*)shift->data;
#pragma omp parallel for num_threads(get_num_threads(e_num)) shared(scale_tmp,shift_tmp,tmp)
    for(i=0;i<e_num;++i){
        size_t ri=e_num/width;
        size_t wi=e_num%width;
        float tmp_sm=.0;
        float tmp2_sm=.0;
        for(int j=0;j<batch;++j){
            tmp_sm+=tmp[j*e_num+i];
            tmp2_sm+=tmp[j*e_num+i]*tmp[j*e_num+i];
        }
        float x_mean=tmp_sm/batch;
        float x_std=tmp2_sm/batch-x_mean*x_mean;
        // cout<<"x_mean:"<<x_mean<<";x_std:"<<x_std<<endl;
        for(int j=0;j<batch;++j){
            tmp[j*e_num+i]=scale_tmp[i]*(tmp[j*e_num+i]-x_mean)/sqrt(x_std+eps)+shift_tmp[i];
        }
    }
    return ret;
}

Conv2D::Conv2D(size_t units,Dim kernel_size,size_t stride,size_t padding,string name)
:Layer(name),units(units),kernel_size(kernel_size),padding(padding),stride(stride){
    if(units<=0){
        throw "units<=0 Error!!!";
    }

    for(int i=0;i<units;++i){
        kernels.push_back( F::randn_tensor(kernel_size));
    }
}
shared_ptr<Tensor> Conv2D::forward(shared_ptr<Tensor>input){
    if(input->dim.size()!=3){
        throw "input tensor shape is Error!!!";
    }
    if(input->dtype!=DType::Float32){
        throw "input dtype is not float32 error!!!";
    }
    size_t batch = input->dim.shape()[0];
    size_t height = input->dim.shape()[1];
    size_t width = input->dim.shape()[2];
    size_t tn=1;


    size_t kernel_height = kernel_size.shape()[0];
    size_t kernel_width = kernel_size.shape()[1];
    size_t new_height = (height+2*padding-kernel_height)/stride+1;
    size_t new_width = (width+2*padding-kernel_width)/stride+1;
    
    auto ret =  F::zero_tensor(Dim(batch,units,new_height,new_width));
    float*data=(float*)input->data;
    // cout<<"hxw:"<<new_height<<"x"<<new_width<<endl;
    size_t e_num=new_height*new_width;
    float*buff=(float*)ret->data;
#pragma omp parallel for num_threads(get_num_threads(e_num)) shared(data,buff,kernels)
    for(int i=0;i<new_height*new_width;++i){
        //data->ret
        size_t ri = i/new_width; // 1
        size_t ci = i%new_width; //1
        int p_ri = ri*stride - padding;
        int p_ci = ci*stride - padding;
        for(int b=0;b<batch;++b){
            for(int k=0;k<units;++k){
                float sm=.0;
                for(int ki=0;ki<kernel_height;++ki){
                    if(p_ri+ki<0 || p_ri+ki>=height){
                        continue;
                    }
                    for(int kj=0;kj<kernel_width;++kj){
                        if(p_ci+kj<0 || p_ci+kj>=width){
                            continue;
                        }
                        sm+=((float*)kernels[k]->data)[ki*kernel_width+kj]*data[(p_ri+ki)*width+p_ci+kj];
                    }
                }
                buff[b*units*new_height*new_width+k*new_height*new_width+i]=sm/(kernel_width*kernel_height);
            }
        }
    }
    return ret;
}