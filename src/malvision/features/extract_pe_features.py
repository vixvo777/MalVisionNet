"""
Extract PE static features for LightGBM baseline.
"""
import pefile
import pandas as pd
import numpy as np
from pathlib import Path
from ..logger import logger

def extract_features(file_path: Path):
    try:
        pe = pefile.PE(file_path)
        features = {
            "section_count": pe.FILE_HEADER.NumberOfSections,
            "image_size": pe.OPTIONAL_HEADER.SizeOfImage,
            "import_count": len(pe.imports) if pe.imports else 0,
            "export_count": len(pe.exports) if pe.exports else 0,
            "machine": pe.FILE_HEADER.Machine,
        }
        return features
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        return None

def build_feature_table(manifest_path: Path):
    df = pd.read_csv(manifest_path)
    feature_list = []

    for _, row in df.iterrows():
        # Assume path is in manifest or infer it
        # This is a placeholder; needs actual file path logic
        file_path = Path(row['sha256']) # Adjust logic
        feats = extract_features(file_path)
        if feats:
            feats['sample_id'] = row['sample_id']
            feature_list.append(feats)

    return pd.DataFrame(feature_list)
