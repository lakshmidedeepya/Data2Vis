#!/bin/bash
#SBATCH -A saigunda
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH -w gnode090
#SBATCH --output=op_file1.txt

module load u22/cuda/12.9
module load u22/cudnn/9.1.0-cuda-12
echo "Running S_Data2Vis_updated.py with 10 tasks and GPU support"

python3 S_Data2Vis_updated.py > op_file_v2_2.txt