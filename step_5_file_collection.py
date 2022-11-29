import pandas as pd

from step_2_generate_excel_and_modify_sign import get_dataframe, output_best_dir
from step_4_generate_images import iter_identifiers


def set_the_mapping_between_names():
    df: pd.DataFrame = get_dataframe(output_best_dir)
    df = df.sort_values('accounting').reset_index(drop=True)
    df['alpha_name'] = [f'Alpha_XYF_{i + 101}' for i in df.index]
    # noinspection PyTypeChecker
    df.to_csv(output_best_dir / 'data.csv', index=None)
    return df


def copy_all_python_files(df: pd.DataFrame):
    for identifier, data in iter_identifiers(df):
        print(identifier)


def main():
    df = set_the_mapping_between_names()
    copy_all_python_files(df)


if __name__ == '__main__':
    main()
