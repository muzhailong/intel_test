#include<iostream>
#include<omp.h>
#include "layers.h"
#include "component.h"
#include<memory>
#include<random>
#include<ctime>
#include<chrono>
using namespace std;

void print(shared_ptr<Tensor>input,bool only_shape=false){
    float*buff=(float*)input->data;
    if(only_shape){
        cout<<"(";
        for(int t : input->dim.shape()){
            cout<<t<<",";
        }
        cout<<")"<<endl;
    }else{
        for(int i=0;i<input->dim.elements_size();++i){
            cout<<buff[i]<<",";
        }
    }
    cout<<endl;
}

int main(){
    auto input=F::randn_tensor(Dim(64,128,128));
    Relu relu_obj;
    BatchNorm2D bn(Dim(128,128));
    Conv2D conv2d(64,Dim(3,3),1,0);

#ifdef _OPENMP
    double st=omp_get_wtime();
#else
    auto st=chrono::system_clock::now();
#endif

    cout<<"input:"<<endl;
    print(input,true);
    
    auto h1=relu_obj.forward(input);
    cout<<"relu:"<<endl;
    print(h1,true);

    
    auto h2=bn.forward(h1);
    cout<<"bn:"<<endl;
    print(h2,true);

    
    auto h3=conv2d.forward(h2);
    cout<<"conv2d:"<<endl;
    print(h3,true);
    
#ifdef _OPENMP
    double ed=omp_get_wtime();
    cout<<"run time:"<<float(ed-st)<<"s"<<endl;
#else
    auto ed=chrono::system_clock::now();
    auto elapsed = chrono::duration_cast<chrono::milliseconds>(ed - st);
    cout<<"run time:"<<elapsed.count()/1000.0 <<"s" <<endl;
#endif
    return 0;
}