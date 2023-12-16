# LatticeSurgeryCCZStateExample

We used this library in [our paper](https://arxiv.org/abs/2309.09893) to determine the number of malicious fault pairs in a lattice-surgery-based $\left \vert \mathrm{CCZ} \right \rangle$ state preparation at distance two.

## How To Replicate The Paper

To reproduce the result, run the following in the julia REPL from this directory:

```julia
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.8.5 (2023-01-08)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

julia> import LatticeSurgeryCCZStateExample as LS

julia> n_fault_pairs = LS.malicious_fault_pairs(LS.lattice_surgery_circuit())
```

Be warned, this takes a few days.