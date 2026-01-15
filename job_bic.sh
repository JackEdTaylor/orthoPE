#!/bin/bash
#SBATCH --job-name=orthope
#SBATCH --partition=96GBLppc,128GBLppc,256GBLppc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=896
#SBATCH --mem-per-cpu=2000
#SBATCH --time=7-00:00:00

# activate the conda env
module load conda
conda activate otlettersimppc
echo conda environment:
echo $CONDA_DEFAULT_ENV
echo ------------------

# limit threads usage for parallelisation
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# estimate models
python pipelines.py
