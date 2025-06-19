#!/bin/bash
#SBATCH --job-name=poisson_exp
#SBATCH --output=logs/poisson_exp_%j.out
#SBATCH --error=logs/poisson_exp_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=barnard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=anna.kurova@tu-dresden.de

# ------------------------ SETUP ------------------------

WSNAME=poisson_$SLURM_JOB_ID
export WSDIR=$(ws_allocate --filesystem horse --name "$WSNAME" --duration 2)
test -z "$WSDIR" && echo "Error: Failed to allocate workspace." && exit 1
echo "Allocated workspace: $WSDIR"

rsync -av \
  --exclude="models_notebooks/" \
  --exclude="logs/" \
  --exclude="run_results*/" \
  --exclude="results/" \
  --exclude=".git" \
  --exclude=".*" \
  --exclude="plots" \
  --exclude="*.sh" \
  --exclude="*.pth" \
  --exclude="*.md" \
  --exclude="helpers/" \
  "$HOME/score_matching/" \
  "$WSDIR/"
cd "$WSDIR" || { echo "Failed to cd into $WSDIR"; exit 1; }
echo "Working inside $WSDIR"

python -m venv .venv
source .venv/bin/activate

export TMPDIR="$WSDIR/tmp_pip"
export PIP_CACHE_DIR="$WSDIR/pip_cache"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

pip install --upgrade pip
pip install --cache-dir="$PIP_CACHE_DIR" -r requirements.txt

RUN_RESULTS_DIR="$HOME/score_matching/run_results_statistical_analysis_distance_weight"
mkdir -p "$RUN_RESULTS_DIR"
echo "Results will be stored in $RUN_RESULTS_DIR"

# ------------------------ EXPERIMENT LOOP ------------------------

SCRIPT_NAME="poisson_experiment"
NUM_JOBS=10

for i in $(seq 1 $NUM_JOBS); do
  (
    echo "Running iteration $i"

    WORK_DIR="workspace_$i"
    mkdir -p "$WORK_DIR"

    CONFIG_FILE="$WORK_DIR/configs/config_${SCRIPT_NAME}_iteration_${i}.json"
    mkdir -p "$WORK_DIR/configs"
    cp "configs/config_${SCRIPT_NAME}_${i}.json" "$CONFIG_FILE"
    sed -i "s/\"region\": \"[^\"]*\"/\"region\": \"region_${i}\"/" "$CONFIG_FILE"

    python "src/run_script.py" \
      --script "$SCRIPT_NAME" \
      --config "$CONFIG_FILE" \
      --workspace "$WORK_DIR"

    if [ -d "$WORK_DIR/results" ]; then
      DEST="$RUN_RESULTS_DIR/results_${SCRIPT_NAME}_job_${SLURM_JOB_ID}_iteration_$i"
      cp -r "$WORK_DIR/results/" "$DEST"
      echo "Copied results for iteration $i to $DEST"
    else
      echo "No results directory found for iteration $i"
    fi

    rm -rf "$WORK_DIR"
    echo "Cleaned up workspace for iteration $i"
  ) &
done

wait
echo "All parallel runs completed."

# ------------------------ CLEANUP ------------------------

rm -rf "$TMPDIR" "$PIP_CACHE_DIR"
echo "Temporary pip/cache directories cleaned up."

ws_release -F horse "$WSNAME"
echo "Workspace $WSNAME released successfully."