import argparse
from pathlib import Path
from train import train_dysat

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
p.add_argument("--data_dir", default="/input")
p.add_argument("--out_dir", default="/output")
p.add_argument("--num_features", type=int, default=128)
p.add_argument("--structural_layer_config", type=int, default=128)
p.add_argument("--structural_head_config", type=int, default=16)
p.add_argument("--temporal_layer_config", type=int, default=128)
p.add_argument("--temporal_head_config", type=int, default=16)
p.add_argument("--spatial_drop", type=float, default=0.1)
p.add_argument("--temporal_drop", type=float, default=0.5)
p.add_argument("--learning_rate", type=float, default=0.001)
p.add_argument("--weight_decay", type=float, default=5e-4)
p.add_argument("--max_gradient_norm", type=float, default=1.0)
p.add_argument("--neg_sample_size", type=int, default=10)
p.add_argument("--neg_weight", type=float, default=1.0)
p.add_argument("--max_positive", type=int, default=10)
p.add_argument("--epochs", type=int, default=150)
p.add_argument("--batch_size", type=int, default=256)
p.add_argument("--patience", type=int, default=10)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

train_dysat(a.dataset, Path(a.data_dir), Path(a.out_dir),
            num_features=a.num_features,
            structural_layer_config=(a.structural_layer_config,),
            structural_head_config=(a.structural_head_config,),
            temporal_layer_config=a.temporal_layer_config,
            temporal_head_config=a.temporal_head_config,
            spatial_drop=a.spatial_drop, temporal_drop=a.temporal_drop,
            learning_rate=a.learning_rate, weight_decay=a.weight_decay,
            max_gradient_norm=a.max_gradient_norm,
            neg_sample_size=a.neg_sample_size, neg_weight=a.neg_weight,
            max_positive=a.max_positive,
            epochs=a.epochs, batch_size=a.batch_size,
            patience=a.patience, seed=a.seed)