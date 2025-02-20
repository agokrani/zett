import pandas as pd
import os
import glob

# List of languages to process
# LANGUAGES = [
#     "en", 
#     "zh", 
#     "cpp",
#     "go",
#     "java",
#     "javascript",
#     "python",
#     "github-issues-filtered-structured"
# ]
LANGUAGES = None

def main():
    # Path to data folder
    data_folder = "data/train"
    output_file = "data/train.txt"
    
    try:
        all_files = []
        
        # If LANGUAGES is None or empty, get all parquet files from the directory
        if not LANGUAGES:
            all_files = glob.glob(os.path.join(data_folder, "*.parquet"))
        else:
            # Get parquet files for each specified language
            for lang in LANGUAGES:
                lang_files = glob.glob(os.path.join(data_folder, f"{lang}*.parquet"))
                all_files.extend(lang_files)
            
        if not all_files:
            print(f"No parquet files found in {data_folder}")
            return
            
        print(f"Found {len(all_files)} parquet files")
        
        # Process each file and write to train.txt
        with open(output_file, 'w', encoding='utf-8') as f:
            for parquet_file in all_files:
                print(f"Processing {parquet_file}")
                df = pd.read_parquet(parquet_file)
                
                if 'text' not in df.columns:
                    print(f"Warning: {parquet_file} does not contain 'text' column. Skipping...")
                    continue
                    
                # Write each text entry to the file
                for text in df['text']:
                    if pd.notna(text):  # Check for non-null values
                        f.write(text.strip() + '\n')
                        
        print(f"Successfully created {output_file}")

    except Exception as e:
        print(f"Error processing files: {e}")

if __name__ == "__main__":
    main()