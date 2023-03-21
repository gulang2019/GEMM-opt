# The GEMM optimization practice 

1. Preliminary;
- g++ 17;
- avx2, avx256, avx512
- git

> It is recommended to use PKU's public server to run the program. See more at https://hpc.pku.edu.cn/guide.html

2. Build the workspace;
```bash 
git clone git@github.com:gulang2019/GEMM-opt.git
```

3. Implement your optimization in dgemm/dgemm.cpp

4. Look at / Run the sample project in optimization/, see `optimization/dgemm-solution-guide.md` for more information.