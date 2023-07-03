#!/bin/sh
#SBATCH --partition=GPUQ 
#SBATCH --account=ie-idi
#SBATCH --mem=192GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4     
#SBATCH --cpus-per-task=4 
#SBATCH --job-name=UTD-I
#SBATCH --time=50:00:00 
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
module load cuDNN/8.2.1.32-CUDA-11.3.1
module load Anaconda3/2020.07
module load fosscuda/2020b 
module laod TensorFlow/2.6.0-foss-2021a-CUDA-11.3.1

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

sleap-track "channel1.mp4" -m "models/230209_221559.centroid" -m "models/230209_221559.centered_instance" --batch_size 1 --tracking.tracker "simple" --tracking.target_instance_count 7   
sleap-track "channel2.mp4" -m "models/230209_221559.centroid" -m "models/230209_221559.centered_instance" --batch_size 1 --tracking.tracker "simple" --tracking.target_instance_count 14   
sleap-track "channel3.mp4" -m "models/230209_221559.centroid" -m "models/230209_221559.centered_instance" --batch_size 1 --tracking.tracker "simple" --tracking.target_instance_count 19   
sleap-track "channel4.mp4" -m "models/230209_221559.centroid" -m "models/230209_221559.centered_instance" --batch_size 1 --tracking.tracker "simple" --tracking.target_instance_count 28   