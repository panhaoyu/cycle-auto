import pandas as pd

from step_2_generate_excel_and_modify_sign import get_dataframe, output_best_dir


def main():
    df: pd.DataFrame = get_dataframe(output_best_dir)
    df = df.sort_values('accounting').reset_index(drop=True)
    df['alpha_name'] = [f'Alpha_XYF_{i + 101}' for i in df.index]
    df.to_csv(output_best_dir / 'data.csv', index=None)


if __name__ == '__main__':
    main()
