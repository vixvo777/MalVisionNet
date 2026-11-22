"""
Generate dataset splits (random, temporal, unseen-family).
"""
import pandas as pd
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
import random

from . import logger

class Splitter:
    def __init__(self, config_path="configs/data.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def create_splits(self, manifest_path: Path):
        df = pd.read_csv(manifest_path)
        logger.info(f"Loaded manifest with {len(df)} samples")

        # 1. Random/Grouped Split (Reference)
        train, test = train_test_split(df, test_size=0.3, random_state=42)
        val, test = train_test_split(test, test_size=0.5, random_state=42)

        df.loc[train.index, 'split'] = 'train'
        df.loc[val.index, 'split'] = 'val'
        df.loc[test.index, 'split'] = 'test'

        # 2. Temporal Split (for malware data)
        # Filter malware samples
        malware_df = df[df['label'] == 'malicious'].copy()

        if not malware_df.empty:
            # Sort by first_seen if available, else by sample_id
            if 'first_seen' in malware_df.columns:
                malware_df = malware_df.sort_values('first_seen')
            else:
                malware_df = malware_df.sample(frac=1, random_state=42)

            # Split temporally: early = train, mid = val, late = test
            total_samples = len(malware_df)
            train_size = int(total_samples * 0.7)
            val_size = int(total_samples * 0.15)

            malware_df.iloc[:train_size, df.columns.get_loc('split')] = 'train'
            malware_df.iloc[train_size:train_size+val_size, df.columns.get_loc('split')] = 'val'
            malware_df.iloc[train_size+val_size:, df.columns.get_loc('split')] = 'test'

            # Update original df
            df.loc[malware_df.index] = malware_df.values

        # 3. Unseen-Family Split
        # Identify all families present in the dataset
        if 'family' in df.columns:
            all_families = df['family'].dropna().unique()

            # Select holdout families (50% of families)
            if len(all_families) > 1:
                holdout_families = random.sample(list(all_families),
                                               k=int(len(all_families) * 0.5))

                # Assign holdout families to test split
                family_mask = df['family'].isin(holdout_families)
                df.loc[family_mask, 'split_family'] = 'test_family'
                df.loc[~family_mask & (df['split'].isnull() | df['split'] == 'test'),
                       'split_family'] = 'train_family'

                # If no split column exists yet, create it
                if 'split' not in df.columns:
                    df['split'] = 'train'

                # Update main split: holdout families to test, others to train/val according to main split
                df.loc[family_mask, 'split'] = 'test_family'
                df.loc[~family_mask, 'split'] = df.loc[~family_mask, 'split'].fillna('train')

        output_path = manifest_path.with_name("malvision_manifest_split.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"Splits saved to {output_path}")

        # Create split summary report
        summary_path = manifest_path.parent / "split_summary.json"
        summary = {
            'random_split': {
                'train_count': len(df[df['split'] == 'train']),
                'val_count': len(df[df['split'] == 'val']),
                'test_count': len(df[df['split'] == 'test'])
            }
        }

        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)

        return output_path

if __name__ == "__main__":
    # Example usage
    splitter = Splitter()
    splitter.create_splits(Path("data/manifests/malvision_manifest.csv"))