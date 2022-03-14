# Intel 面试题
## Question
<pre>
The Convolution, BatchNorm and ReLU is a basic unit of CNN.

In this test you need write the function of these three parts by C/C++.

Forward Only; Backward is a PLUS;
Verify the results by test case and calculate the computation efficiency
 Select at least one of below two methods to improve performance

Fuse the Conv + BatchNorm + ReLU into one function to reduce the memory access
Reference: https://sc18.supercomputing.org/proceedings/tech_poster/poster_files/post155s2-file2.pdf

Parallel with popular parallel technique. e.g. OpenMP, TBB
Reference: https://computing.llnl.gov/tutorials/openMP/
</pre>
<pre>
 calculate the computation efficiency:不是很理解，纯粹单线程（非向量化指令）执行，计算效率就和计算的次数相关。并行优化中，多个线程执行，相当于同一时间执行了更多的指令（多个core上）执行效率近似和线程数目正相关（这个test中线程之间没有数据竞争，有一部分线程切换的开销）
</pre>
## Usage
1. 并行执行:
    直接make
2. 单线程执行：
    删除Makefile中的-fopenmp 然后直接make
<p>然后,直接执行./test进行单元测试,或者直接执行./main执行example</p>

## Performance
在Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz   2.00 GHz，16G memory，WSL下进行测试,采用（64,128,128)的输入尺寸，64个3x3的kernel进行性能测试
</br>
![avatar](https://github.com/muzhailong/intel_test/raw/master/imgs/1.jpg?raw=true)
