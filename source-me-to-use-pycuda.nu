# you need cuda and to source these:
load-env { CUDA_PATH: '/opt/cuda', NVCC_CCBIN: '/usr/bin/g++-13' }
$env.PATH = $env.PATH | append [ '/opt/cuda/bin' '/opt/cuda/nsight_compute' '/opt/cuda/nsight_systems/bin' ] 
