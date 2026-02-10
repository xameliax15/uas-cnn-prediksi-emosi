import shutil
import os
import zipfile

def zip_project(output_filename):
    # Get current directory
    source_dir = os.getcwd()
    
    # Exclude patterns
    exclude_dirs = {'.git', '.idea', '__pycache__', 'venv', 'env', 'node_modules', '.gemini'}
    exclude_files = {'.DS_Store', 'zip_project.py', output_filename, 'diagnose_model.py', 
                     'diagnose_model_v2.py', 'augment_and_train.py'}
    
    print(f"Zipping project from {source_dir} to {output_filename}...")
    
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            # Exclude directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                if file in exclude_files:
                    continue
                if file.endswith('.pyc') or file.endswith('.pyo') or file.endswith('.pyd'):
                    continue
                    
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_dir)
                
                print(f"Adding: {arcname}")
                zipf.write(file_path, arcname)
                
    print(f"\n✅ Build complete! File saved as: {output_filename}")
    print(f"Total size: {os.path.getsize(output_filename) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    zip_project('emotion_classification_app.zip')
