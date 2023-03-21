# Optimization Techniques for DGEMM
The following techniques are applicable to many problems.


## Compilelation 
```bash 
g++ ./dgemm-answer.cpp -o dgemm -march=native -fopenmp -O2 -std=c++17
```

Run the program. 
```bash
./dgemm 
```
> If you are on a server that uses time-sharing system (e.g. wmjx2), you need to provide a `.slurm` file to describe your execution.
An example is written in `run.slurm`. Upon running, use `sbatch ./run.slurm` and check result in ./slurm-[number].out. 


## ILP
### Compiler Optimization Flags
- Ofast
- funroll-loops
- Profiling

## DLP
### AVX
- AVX2/AVX512
  - Are *ymm* operations always faster than *xmm* counterparts? If not, why?
    - Hint: consider memory.
- FMA
  - What is the advantage of FMA?
    - Speed: one cycle less
    - Accuracy: one rounding less

Aside: Can *gcc* generate *AVX* and *FMA* instructions automatically? What flags are required?
- *O3* and higher (*Ofast*)
- Difference between *O3* and *Ofast*
  - E.g., sometimes reordering **FP operations** is allowable. We know that **FP operations** are not associative, which is a great obstacle for parallelism.

### One Step Further: CUDA
- Why is GPU excel in this kind of computation?
  - The answer can be found in another document.

## TLP
- For simplicity, *OpenMP*.

## Cache
### Loop Interchange
- *ijk* vs *ikj*.
- Why is *ikj* faster than *kij* by a large margin even if correctness is ignored in the multi-threading environment?
  - Hint: Cache coherence. See **CAAQA Chap 5**.
### Cache Blocking
- Compute order.
- Storage order (tiling), e.g., transform *2D array* into *4D array*.
