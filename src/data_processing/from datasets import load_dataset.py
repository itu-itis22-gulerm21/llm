from datasets import load_dataset
import pandas as pd

# Parquet dosyalarını oku
dataset = load_dataset(
    "parquet",
    data_dir="data/raw_data/druggpt_binding_affinity"
)

# İlk birkaç örneği göster
print("Dataset keys:", dataset.keys())
print("\nFirst 3 examples:")
print(dataset['train'][:3])

# Kolonları göster
print("\nColumns:", dataset['train'].column_names)

# Veri tipi ve boyut
print(f"\nTotal samples: {len(dataset['train'])}")

# Örnek bir satır detaylı
example = dataset['train'][0]
print("\nDetailed example:")
for key, value in example.items():
    print(f"{key}: {value[:100] if isinstance(value, str) else value}")