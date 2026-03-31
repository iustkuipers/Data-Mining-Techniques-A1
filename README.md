# Data Mining Techniques A1

This project processes the smartphone mood dataset in two steps:

1. Data exploration reads the raw file from `data/raw/dataset_mood_smartphone.csv` and writes an intermediate dataset to `data/intermediate/df_wide.csv`.
2. Data cleaning reads the intermediate dataset and writes the cleaned output to `data/clean/df_clean.csv`.

## Run

From the project root, run:

```bash
python main.py
```

The pipeline will execute data exploration first and then data cleaning.