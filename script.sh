#!/bin/bash
#SBATCH --job-name=comp_analysis
#SBATCH --output=logs/comp_analysis_%j.out
#SBATCH --error=logs/comp_analysis_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=barnard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anna.kurova@tu-dresden.de

# Load Python module if needed
# module load Python/3.12.3
# module spider Python/3.12.3

# Define script argument
SCRIPT_NAME="poisson_experiment"

# Set up unique workspace directory
WORKDIR="/data/horse/ws/${USER}/score_matching_job_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"

# Sync project to workspace
rsync -av --exclude="models_notebooks/" --exclude="logs/" --exclude="results_*/" "$HOME/score_matching/" "$WORKDIR/"
echo "Working in $WORKDIR" || { echo "Failed to cd into $WORKDIR"; exit 1; }

# Move to project folder
cd "$WORKDIR"

# Activate virtual environment
source .venv/bin/activate

# Create temp and cache directories
export TMPDIR="$HOME/score_matching/tmp_pip"
export PIP_CACHE_DIR="$HOME/score_matching/pip_cache"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

# Install packages using pip with controlled temp/cache dirs
pip install --upgrade pip
pip install --cache-dir="$PIP_CACHE_DIR" -r requirements.txt

# Run script with argument
python src/run_script.py --script "$SCRIPT_NAME"

# Copy results back and clean up
DEST="$HOME/score_matching/results_${SCRIPT_NAME}_job_${SLURM_JOB_ID}"
cp -r results "$DEST"
echo "Results copied back to $DEST"

# Remove results from workspace
rm -rf results
echo "Results directory removed from workspace."