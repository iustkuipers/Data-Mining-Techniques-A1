from pathlib import Path

from modules.data.data_clean import run as run_data_cleaning
from modules.data.data_exploration import run as run_data_exploration


def main():
	project_root = Path(__file__).resolve().parent
	raw_input_path = project_root / "data" / "raw" / "dataset_mood_smartphone.csv"
	intermediate_output_path = project_root / "data" / "intermediate" / "df_wide.csv"

	#data exploration
	run_data_exploration(raw_input_path)

	#1 data cleaning
	run_data_cleaning(intermediate_output_path)


if __name__ == "__main__":
	main()
