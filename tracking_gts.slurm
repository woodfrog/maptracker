#!/bin/bash
#SBATCH --job-name=tracking_gts
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=450G
#SBATCH --gres=gpu:0
#SBATCH --time=4:00:00
#SBATCH --output=logs/tracking_gts_%j.out
#SBATCH --error=logs/tracking_gts_%j.err
#SBATCH --mail-type=end
#SBATCH --mail-user=ricky.chen@princeton.edu

# Set up environment
cd /n/fs/dynamicbias/tracker
eval "$(conda shell.bash hook)"
conda activate ./env-maptracker

# Prevent thread explosion
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1



#### FILE TRANSFER ####
DIR_NAME="/scratch/rc5898/tracker"
TAR_PATH="/scratch/rc5898/tracker.tar"

# Check if the tracker folder exists. If not, we need to handle the tar.
if [ ! -d "$DIR_NAME" ]; then
    echo "Tracker directory not found."

    # 1. Check if we even have the tar file; if not, download it
    if [ ! -f "$TAR_PATH" ]; then
        echo "Tar file not found. Downloading from login node..."
        expect << EOF
set timeout -1
set send_slow {1 .1}
spawn rsync -avP "rc5898@neuronic.cs.princeton.edu:$TAR_PATH" "/scratch/rc5898/"
expect "Passcode or option (1-3):"
sleep 3
send -s "1\r"
expect eof
EOF
    fi

    echo "Uncompressing $TAR_PATH..."
    tar -xf "$TAR_PATH" -C /scratch/rc5898/
    echo "Uncompress complete."
else
    echo "Directory $DIR_NAME already exists. Skipping transfer."
fi


echo "Starting run..."
ln -sfn /scratch/rc5898/tracker/datasets /n/fs/dynamicbias/tracker/datasets
python tools/tracking/prepare_gt_tracks.py \
    plugin/configs/maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py \
    --out-dir tracking_gts/nuscenes

echo "Run completed!"
