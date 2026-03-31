# Data Mining Techniques A1

Assignment 1 for Data Mining Techniques at VU Amsterdam, advanced track: mood prediction from smartphone sensor data.

## Setup

Install the required packages:

```bash
pip install pandas numpy matplotlib seaborn scipy
```

Place `dataset_mood_smartphone.csv` in `data/raw/`.

## Run

From the project root, run:

```bash
python main.py
```

To run each step separately:

```bash
python modules/data/data_exploration.py
python modules/data/data_clean.py
```

## Structure

```text
.
├── main.py
├── data/
│   ├── raw/
│   │   └── dataset_mood_smartphone.csv
│   ├── intermediate/
│   │   └── df_wide.csv
│   └── clean/
│       └── df_clean.csv
├── modules/
│   └── data/
│       ├── data_exploration.py
│       └── data_clean.py
└── plots/
	└── data_exploration/
```

## Pipeline

`main.py` runs the full pipeline in this order:

1. `data_exploration.py` reads the raw long-format dataset, performs EDA, pivots the data to daily wide format, and saves `data/intermediate/df_wide.csv`.
2. `data_clean.py` reads `data/intermediate/df_wide.csv`, applies the cleaning steps, and saves `data/clean/df_clean.csv`.

## Cleaning Decisions

- Negative durations are removed as hard errors.
- Heavy-tailed `appCat` columns are Winsorized at the 1st and 99th percentiles.
- `call` and `sms` are re-encoded as daily binary presence flags.
- Missing `appCat` values are filled with `0`, treating missing as not used that day.
- `mood`, `circumplex.arousal`, `circumplex.valence`, `screen`, and `activity` are imputed per subject with a 7-day rolling mean.
- Gaps longer than 3 days are imputed but marked with `{variable}_synthetic = 1`.