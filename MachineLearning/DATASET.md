# Dataset

- **Name**: Crop Recommendation Dataset
- **Source**: [Kaggle - Crop Recommendation Dataset (atharvaingle)](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
- **Local File**: `data/soil_quality_crop_recommendation_dataset.csv`
- **Size**: 2,200 rows, 8 columns, 22 crop classes (`label`)

## Schema

| Column | Type | Description |
| --- | --- | --- |
| `N` | float | Nitrogen content ratio in soil |
| `P` | float | Phosphorus content ratio in soil |
| `K` | float | Potassium content ratio in soil |
| `temperature` | float | Ambient temperature (°C) |
| `humidity` | float | Relative humidity (%) |
| `ph` | float | Soil pH value |
| `rainfall` | float | Rainfall (mm) |
| `label` | string | Target crop class (22 categories) |
