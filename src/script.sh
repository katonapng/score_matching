#!/bin/bash
#SBATCH --job-name=comp_analysis
#SBATCH --output=logs/comp_analysis_%j.out
#SBATCH --error=logs/comp_analysis_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=a.kurova19@gmail.com

cd ...
source .venv/Scripts/activate
python run_script.py --script comparative_analysis