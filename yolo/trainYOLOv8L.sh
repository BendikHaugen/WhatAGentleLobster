#!/bin/sh
#SBATCH --partition=GPUQ 
#SBATCH --account=ie-idi
#SBATCH --mem=1031GB
#SBATCH --nodes=2
#SBATCH --output=yolov8L.out
#SBATCH --ntasks-per-node=4     
#SBATCH --cpus-per-task=4 
#SBATCH --job-name=L
#SBATCH --time=200:00:00 
#SBATCH --export=ALL 
#SBATCH --mail-user=Bendik_haugen@hotmail.com
#SBATCH --gres=gpu:1

echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"
echo "Total of GPUS: $CUDA_VISIBLE_DEVICES"

nvidia-smi
nvidia-smi nvlink -s
nvidia-smi topo -m
module purge


__conda_setup="$('/cluster/apps/eb/software/Anaconda3/2020.07/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
   if [ -f "/cluster/apps/eb/software/Anaconda3/2020.07/etc/profile.d/conda.sh" ]; then
      . "/cluster/apps/eb/software/Anaconda3/2020.07/etc/profile.d/conda.sh"
   else
      export PATH="/cluster/apps/eb/software/Anaconda3/2020.07/bin:$PATH"
   fi
fi
unset __conda_setup

ipython -c "run yolov8Ltraining.ipynb"


