#!/bin/bash
#SBATCH -J JobLong               # sensible name for the job
#SBATCH -N 2                    # Allocate 2 nodes for the job
#SBATCH --ntasks-per-node=1     # 1 task per node
#SBATCH -c 20
#SBATCH -t 35:00:00             # Upper time limit for the job
#SBATCH -p GPUQ


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
module load cuDNN/8.2.1.32-CUDA-11.3.1
module load Anaconda3/2020.07
module load fosscuda/2020b 
module laod TensorFlow/2.6.0-foss-2021a-CUDA-11.3.1
module load intel/2019a
module load SciPy-bundle/2019.03

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

conda activate sleap

sleap-train centroid.json train_with_track_id.pkg.slp
sleap-train centered_instance.json train_with_track_id.pkg.slp