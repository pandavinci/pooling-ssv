#!/usr/bin/env python3
"""
Script generator for parallel feature extraction.
Creates 10 scripts for each dataset split (train, dev, test) to parallelize computation.
"""

import os
import argparse

def create_extraction_script(split, chunk_id, feature_type, total_chunks=10):
    """Create a single extraction script for a specific chunk."""
    script_content = f"""#!/bin/bash
#PBS -N extract_{split}_{feature_type}_{chunk_id}
#PBS -l select=1:ncpus=1:mem=16gb:scratch_local=50gb
#PBS -l walltime=24:00:00
#PBS -m ae

#### VARIABLES ####
export OMP_NUM_THREADS=$PBS_NUM_PPN
export TASK="asv-zoo"
export DATASET_PATH=datasets/slt-sstc
export HOME=/storage/brno2/home/${{USER}}
export SRCDIR=/storage/brno2/home/${{USER}}/${{TASK}}

test -n "$SCRATCHDIR" || {{ echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }}
export TMPDIR=${{SCRATCHDIR}}

echo "${{PBS_JOBID}} is running on node `hostname -f` in a scratch directory ${{SCRATCHDIR}}" >> ${{SRCDIR}}/jobs_info.txt

#### MODULES ####
echo "Loading modules at $(date)"
module add anaconda3/2019.10

echo "Loading environment at $(date)"
cd /storage/plzen1/home/${{USER}}/
source activate .conda/envs/${{TASK}}

echo "Starting program at $(date)"
cd ${{SRCDIR}}

#### FEATURE EXTRACTION ####
echo "Starting feature extraction for {split} split, chunk {chunk_id}, feature type {feature_type}"

# Define paths
CSV_PATH="../datasets/slt-sstc/{split}_list_{chunk_id}.csv"
OUTPUT_PATH="../datasets/slt-sstc/{split}_list_{feature_type}_{chunk_id}.npz"
ROOT_DIR="../datasets/slt-sstc/"

echo "CSV Path: $CSV_PATH"
echo "Output Path: $OUTPUT_PATH"
echo "Using CPU processing"

# Run feature extraction
python3 scripts/extraction/extract_features.py \\
    --csv_path "$CSV_PATH" \\
    --feature_type {feature_type} \\
    --output_path "$OUTPUT_PATH" \\
    --root_dir "$ROOT_DIR" \\
    --device cpu

echo "Feature extraction completed for {split} split, chunk {chunk_id}, feature type {feature_type} at $(date)"
echo "Job finished at $(date)"
clean_scratch
"""
    return script_content

def create_merge_script(split, feature_type, total_chunks=10):
    """Create a script to merge all chunks into a single compressed file."""
    script_content = f"""#!/usr/bin/env python3
\"\"\"
Merge {feature_type} features from {split} split chunks into a single compressed file.
\"\"\"

import numpy as np
import os
from tqdm import tqdm

def merge_features():
    base_dir = "../datasets/slt-sstc/"
    
    all_features = []
    all_file_paths = []
    all_labels = []
    
    # Check if we're dealing with pair or single format
    first_file = os.path.join(base_dir, "{split}_list_{feature_type}_1.npz")
    if not os.path.exists(first_file):
        print(f"Error: First chunk file not found: {{first_file}}")
        return
    
    # Load first file to determine format
    first_data = np.load(first_file, allow_pickle=True)
    is_pair_format = 'source_features' in first_data.files
    
    print(f"Merging {total_chunks} chunks of {feature_type} features for {split} split")
    print(f"Format detected: {{'pair' if is_pair_format else 'single'}}")
    
    if is_pair_format:
        all_source_features = []
        all_target_features = []
        
        for chunk_id in tqdm(range(1, {total_chunks + 1}), desc="Merging chunks"):
            chunk_file = os.path.join(base_dir, f"{split}_list_{feature_type}_{{chunk_id}}.npz")
            
            if os.path.exists(chunk_file):
                data = np.load(chunk_file, allow_pickle=True)
                all_source_features.extend(data['source_features'])
                all_target_features.extend(data['target_features'])
                all_file_paths.extend(data['file_paths'])
                all_labels.extend(data['labels'])
                print(f"Loaded chunk {{chunk_id}}: {{len(data['source_features'])}} samples")
            else:
                print(f"Warning: Chunk file not found: {{chunk_file}}")
        
        # Save merged data
        output_file = os.path.join(base_dir, f"{split}_list_{feature_type}.npz")
        np.savez_compressed(
            output_file,
            source_features=np.array(all_source_features, dtype=object),
            target_features=np.array(all_target_features, dtype=object),
            file_paths=np.array(all_file_paths, dtype=object),
            labels=np.array(all_labels)
        )
        
        print(f"Merged {{len(all_source_features)}} pairs saved to {{output_file}}")
        
    else:
        for chunk_id in tqdm(range(1, {total_chunks + 1}), desc="Merging chunks"):
            chunk_file = os.path.join(base_dir, f"{split}_list_{feature_type}_{{chunk_id}}.npz")
            
            if os.path.exists(chunk_file):
                data = np.load(chunk_file, allow_pickle=True)
                all_features.extend(data['features'])
                all_file_paths.extend(data['file_paths'])
                all_labels.extend(data['labels'])
                print(f"Loaded chunk {{chunk_id}}: {{len(data['features'])}} samples")
            else:
                print(f"Warning: Chunk file not found: {{chunk_file}}")
        
        # Save merged data
        output_file = os.path.join(base_dir, f"{split}_list_{feature_type}.npz")
        np.savez_compressed(
            output_file,
            features=np.array(all_features, dtype=object),
            file_paths=np.array(all_file_paths),
            labels=np.array(all_labels)
        )
        
        print(f"Merged {{len(all_features)}} samples saved to {{output_file}}")

if __name__ == "__main__":
    merge_features()
"""
    return script_content

def create_split_csv_script(total_chunks=10):
    """Create a script to split CSV files into chunks."""
    script_content = f"""#!/usr/bin/env python3
\"\"\"
Split CSV files into {total_chunks} chunks for parallel processing.
\"\"\"

import pandas as pd
import os
import numpy as np

def split_csv_files():
    base_dir = "../datasets/slt-sstc/"
    
    # List of CSV files to split (based on metacentrum environment config)
    csv_files = ["train_list.csv", "dev_trials_50k.csv", "test_trials.csv"]
    split_mapping = {{
        "train_list.csv": "train_list",
        "dev_trials_50k.csv": "dev_list", 
        "test_trials.csv": "test_list"
    }}
    
    for csv_file in csv_files:
        csv_path = os.path.join(base_dir, csv_file)
        
        if not os.path.exists(csv_path):
            print(f"Warning: CSV file not found: {{csv_path}}")
            continue
            
        # Load CSV
        df = pd.read_csv(csv_path)
        print(f"Splitting {{csv_file}} with {{len(df)}} samples into {total_chunks} chunks")
        
        # Split into chunks
        chunk_size = len(df) // {total_chunks}
        remainder = len(df) % {total_chunks}
        
        start_idx = 0
        for chunk_id in range(1, {total_chunks + 1}):
            # Calculate end index for this chunk
            current_chunk_size = chunk_size + (1 if chunk_id <= remainder else 0)
            end_idx = start_idx + current_chunk_size
            
            # Create chunk
            chunk_df = df.iloc[start_idx:end_idx].copy()
            
            # Save chunk with standardized naming
            split_name = split_mapping[csv_file]
            chunk_filename = f"{{split_name}}_{{chunk_id}}.csv"
            chunk_path = os.path.join(base_dir, chunk_filename)
            chunk_df.to_csv(chunk_path, index=False)
            
            print(f"Created {{chunk_filename}}: {{len(chunk_df)}} samples (rows {{start_idx}}-{{end_idx-1}})")
            start_idx = end_idx
    
    print("CSV splitting completed!")
    print("\\nCreated chunks:")
    for csv_file in csv_files:
        split_name = split_mapping[csv_file] 
        for chunk_id in range(1, {total_chunks + 1}):
            print(f"  {{split_name}}_{{chunk_id}}.csv")

if __name__ == "__main__":
    split_csv_files()
"""
    return script_content

def create_master_script(feature_types=['fdlp', 'mel'], total_chunks=10):
    """Create a master script to submit all extraction jobs."""
    script_content = f"""#!/bin/bash
# Master script to submit all feature extraction jobs
# Usage: ./submit_all_jobs.sh [--fast]
# - with the "--fast" argument, the script will qsub all jobs instantly
# - without argument, the script will qsub all jobs with a 63 seconds delay between each job

script_dir=$(dirname "$0")
delay=63

if [ "$1" = "--fast" ]; then
    delay=0
fi

echo "Submitting feature extraction jobs..."

# Submit extraction jobs for each split, chunk, and feature type
"""
    
    for split in ['train', 'dev', 'test']:
        for feature_type in feature_types:
            for chunk_id in range(1, total_chunks + 1):
                script_content += f"""
# Submit {split} {feature_type} chunk {chunk_id}
echo "Queueing extract_{split}_{feature_type}_{chunk_id}.sh"
qsub "$script_dir/extraction/extract_{split}_{feature_type}_{chunk_id}.sh"
if [ "$delay" -gt 0 ]; then
    echo "Sleeping for $delay seconds before the next job"
    sleep "$delay"
fi
"""
    
    script_content += f"""
echo "All extraction jobs submitted!"
echo "Monitor with: qstat -u $USER"
echo ""
echo "After all jobs complete, run merge scripts:"
"""
    
    for split in ['train', 'dev', 'test']:
        for feature_type in feature_types:
            script_content += f"echo \"python3 scripts/merge_{split}_{feature_type}.py\"\n"
    
    return script_content

def main():
    parser = argparse.ArgumentParser(description='Generate parallel feature extraction scripts')
    parser.add_argument('--output_dir', type=str, default='scripts', 
                       help='Output directory for scripts')
    parser.add_argument('--total_chunks', type=int, default=10, 
                       help='Number of chunks per split')
    parser.add_argument('--feature_types', nargs='+', default=['fdlp', 'mel'],
                       help='Feature types to extract')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    extraction_dir = os.path.join(args.output_dir, 'extraction')
    os.makedirs(extraction_dir, exist_ok=True)
    
    splits = ['train', 'dev', 'test']
    
    print(f"Generating scripts for {args.total_chunks} chunks per split")
    print(f"Feature types: {args.feature_types}")
    print(f"Output directory: {args.output_dir}")
    print(f"Extraction scripts directory: {extraction_dir}")
    
    # Generate extraction scripts
    for split in splits:
        for feature_type in args.feature_types:
            for chunk_id in range(1, args.total_chunks + 1):
                script_content = create_extraction_script(split, chunk_id, feature_type, args.total_chunks)
                script_filename = f"extract_{split}_{feature_type}_{chunk_id}.sh"
                script_path = os.path.join(extraction_dir, script_filename)
                
                with open(script_path, 'w') as f:
                    f.write(script_content)
                
                # Make executable
                os.chmod(script_path, 0o755)
                print(f"Created: {script_path}")
    
    # Generate merge scripts
    for split in splits:
        for feature_type in args.feature_types:
            merge_content = create_merge_script(split, feature_type, args.total_chunks)
            merge_filename = f"merge_{split}_{feature_type}.py"
            merge_path = os.path.join(extraction_dir, merge_filename)
            
            with open(merge_path, 'w') as f:
                f.write(merge_content)
            
            # Make executable
            os.chmod(merge_path, 0o755)
            print(f"Created: {merge_path}")
    
    # Generate CSV splitting script
    split_csv_content = create_split_csv_script(args.total_chunks)
    split_csv_path = os.path.join(extraction_dir, "split_csv_files.py")
    with open(split_csv_path, 'w') as f:
        f.write(split_csv_content)
    os.chmod(split_csv_path, 0o755)
    print(f"Created: {split_csv_path}")
    
    # Generate master script
    master_content = create_master_script(args.feature_types, args.total_chunks)
    master_path = os.path.join(args.output_dir, "submit_all_jobs.sh")
    with open(master_path, 'w') as f:
        f.write(master_content)
    os.chmod(master_path, 0o755)
    print(f"Created: {master_path}")
    
    print(f"\nGenerated {len(splits) * len(args.feature_types) * args.total_chunks} extraction scripts")
    print(f"Generated {len(splits) * len(args.feature_types)} merge scripts")
    print("\nUsage:")
    print("1. First split CSV files: python3 scripts/extraction/split_csv_files.py")
    print("2. Submit all jobs: bash scripts/submit_all_jobs.sh [--fast]") 
    print("3. Monitor jobs: qstat -u $USER")
    print("4. After completion, run merge scripts for each split/feature combination:")
    for split in splits:
        for feature_type in args.feature_types:
            print(f"   python3 scripts/extraction/merge_{split}_{feature_type}.py")

if __name__ == "__main__":
    main() 