import os
import argparse
import shutil

# Redirect caches to E: drive to save C: space
HF_CACHE = os.path.abspath("hf_cache")
os.makedirs(HF_CACHE, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE

def export_to_onnx(model_id, output_dir):
    print(f"Exporting {model_id} to ONNX format...")
    print("This will apply O2 optimizations (including basic quantization) for edge devices.")
    
    # Ensure output dir exists
    os.makedirs(output_dir, exist_ok=True)
    
    # We use optimum-cli to export the model for feature extraction
    # The --O2 flag applies ONNX Runtime optimizations
    command = f"optimum-cli export onnx -m {model_id} --task feature-extraction --O2 {output_dir}"
    
    exit_code = os.system(command)
    if exit_code == 0:
        print(f"\nSuccessfully exported to {output_dir}.")
        
        # Check sizes
        total_size = sum(os.path.getsize(os.path.join(dirpath, filename)) 
                         for dirpath, _, filenames in os.walk(output_dir) 
                         for filename in filenames)
        print(f"Total model size on disk: {total_size / (1024 * 1024):.2f} MB")
    else:
        print("\nExport failed. Make sure optimum[exporters] is installed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export HuggingFace model to ONNX")
    parser.add_argument("--model", type=str, default="sentence-transformers/clip-ViT-B-32")
    parser.add_argument("--output", type=str, default="onnx_model")
    args = parser.parse_args()
    
    export_to_onnx(args.model, args.output)
