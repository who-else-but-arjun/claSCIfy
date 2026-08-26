import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
from Mistral7b_Instruct_2 import Doraemon_justification
from Binary_classification import DoraemonBinaryClassifier
from Conference_classification import DoraemonConferenceClassifier

def load_model(model, checkpoint_path, device):
    """Load model weights and, if present, the training-time feature normalization stats
    (feature_weights/feature_mean/feature_std) that must be reapplied identically at inference."""
    # Load the checkpoint file onto the specified target device (CPU or CUDA)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Verify if the saved checkpoint contains the full model state dictionary
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # If the state dictionary is stored directly, load it directly
        model.load_state_dict(checkpoint)

    norm_stats = None
    if isinstance(checkpoint, dict) and 'feature_mean' in checkpoint and 'feature_std' in checkpoint:
        norm_stats = {
            'weights': checkpoint.get('feature_weights'),
            'mean': checkpoint['feature_mean'].to(device),
            'std': checkpoint['feature_std'].to(device),
        }
        if norm_stats['weights'] is not None:
            norm_stats['weights'] = norm_stats['weights'].to(device)
    return model, norm_stats

def normalize_features(raw_features, norm_stats):
    """Apply the exact weight-multiply + normalize steps used at training time.
    Falls back to a warning + no-op if the checkpoint predates saved normalization stats."""
    if norm_stats is None:
        print("[WARNING] Checkpoint has no saved normalization stats - using raw features. "
              "Retrain with the current Binary_classification.py/Conference_classification.py to fix this.")
        return raw_features
    features = raw_features
    if norm_stats['weights'] is not None:
        features = features * norm_stats['weights']
    return (features - norm_stats['mean']) / norm_stats['std']

def process_saved_data(input_dir: Path, output_dir: Path, conference_confidence_threshold: float = 0.4):
    print("[INFO] Initializing processing of saved data...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    text_dir = input_dir / "texts"
    vector_dir = input_dir / "vectors"
    keywords_dir = input_dir / "keywords"

    for dir_path in [text_dir, vector_dir, keywords_dir]:
        if not dir_path.exists():
            raise ValueError(f"Directory not found: {dir_path}")
        
    vector_files = list(vector_dir.glob("*.pt"))
    if not vector_files:
        raise ValueError(f"No vector files found in {vector_dir}")
    print(f"[INFO] Found {len(vector_files)} files to process")

    sample_vector = torch.load(vector_files[0], map_location=device)
    input_dim = sample_vector.shape[0]
    print(f"[INFO] Detected input dimension: {input_dim}")

    try:
        binary_classifier = DoraemonBinaryClassifier(input_dim=input_dim).to(device)
        conference_classifier = DoraemonConferenceClassifier(input_dim=input_dim, num_classes=5).to(device)

        binary_classifier, binary_norm_stats = load_model(binary_classifier, "doraemon_binary_classifier.pt", device)
        conference_classifier, conference_norm_stats = load_model(conference_classifier, "doraemon_conference_classifier.pt", device)

        binary_classifier.eval()
        conference_classifier.eval()
    except Exception as e:
        raise RuntimeError(f"Error loading models: {str(e)}")

    label_map = {0: "CVPR", 1: "TMLR", 2: "KDD", 3: "NEURIPS", 4: "EMNLP"}
    
    print("[INFO] Loading and processing saved data...")
    features_list = []
    file_ids = []
    
    for vector_file in tqdm(vector_files, desc="Loading vectors"):
        try:
            features = torch.load(vector_file, map_location=device)
            features_list.append(features)
            file_ids.append(vector_file.stem)
        except Exception as e:
            print(f"[WARNING] Error loading vector {vector_file}: {str(e)}")
            continue

    print("[INFO] Processing with normalized features...")
    results = []
    
    for idx, file_id in enumerate(tqdm(file_ids, desc="Processing files")):
        try:
            text_file = text_dir / f"{file_id}.json"
            keywords_file = keywords_dir / f"{file_id}.txt"
            
            with open(text_file, 'r') as f:
                parsed_content = json.load(f)
            
            with open(keywords_file, 'r') as f:
                keywords = f.read().splitlines()

            abstract = ""
            conclusion = ""
            for heading, content in parsed_content.items():
                if 'abstract' in heading.lower() or 'introduction' in heading.lower():
                    abstract = content
                elif 'conclusion' in heading.lower() or 'summary' in heading.lower():
                    conclusion = content

            if abstract == "" or conclusion == "":
                for heading, content in parsed_content.items():
                    if 'abstract' in content.lower() or 'introduction' in content.lower():
                        abstract = content
                        break
                for heading, content in parsed_content.items():
                    if 'conclusion' in content.lower() or 'summary' in content.lower():
                        conclusion = content
                        break

            raw_features = features_list[idx]

            with torch.no_grad():
                binary_input = normalize_features(raw_features, binary_norm_stats).unsqueeze(0).to(device)
                binary_pred = binary_classifier(binary_input)
                is_publishable = binary_pred.item() > 0.5

                conference = "na"
                justification = "na"

                if is_publishable:
                    conference_input = normalize_features(raw_features, conference_norm_stats).unsqueeze(0).to(device)
                    conf_logits = conference_classifier(conference_input)
                    conf_probs = torch.softmax(conf_logits, dim=1)
                    conference_id = torch.argmax(conf_probs, dim=1).item()
                    conference_prob = conf_probs[0][conference_id].item()

                    if conference_prob > conference_confidence_threshold:
                        conference = label_map[conference_id]
                        justification = Doraemon_justification(
                            abstract=abstract,
                            conclusion=conclusion,
                            keywords=keywords,
                            conference_name=conference
                        )
                    else:
                        conference = "uncertain"
            
            results.append([file_id, int(is_publishable), conference, justification])
            
        except Exception as e:
            print(f"[WARNING] Error processing results for {file_id}: {str(e)}")
            results.append([file_id, 0, 'error', f'Error: {str(e)}'])

    df = pd.DataFrame(results, columns=['Paper ID', 'Publishable', 'Conference', 'Rationale'])
    df.to_csv(output_dir / "results.csv", index=False)
    print(f"[INFO] Results saved to {output_dir / 'results.csv'}")

if __name__ == "__main__":
    input_dir = Path("Sample")
    output_dir = Path("Sample")
    process_saved_data(input_dir, output_dir)