import shutil

import pandas as pd

from step_1_run_factor_test import base_dir
from step_2_generate_excel_and_modify_sign import get_dataframe, output_best_dir
from step_4_generate_images import iter_identifiers

output_submit_dir = base_dir / 'output-submit'


def set_the_mapping_between_names():
    df: pd.DataFrame = get_dataframe(output_best_dir)
    df = df.sort_values('accounting').reset_index(drop=True)
    df['alpha_name'] = [f'Alpha_XYF_{i + 101}' for i in df.index]
    # noinspection PyTypeChecker
    df.to_csv(output_best_dir / 'data.csv', index=None)
    return df


def copy_all_python_files(df: pd.DataFrame):
    shutil.rmtree(output_submit_dir, ignore_errors=True), output_submit_dir.mkdir()
    for identifier, data in iter_identifiers(df):
        # 复制python文件
        python_files = list((output_best_dir / identifier).glob('*.py'))
        python_files = [p for p in python_files if not p.stem == 'Alpha']
        python_src = python_files[0]
        alpha_name = data['alpha_name']
        accounting = data['accounting']
        python_dst = output_submit_dir / f'{alpha_name}.py'
        shutil.copy(python_src, python_dst)

        # 复制config文件
        config_src = output_best_dir / identifier / 'config.xml'
        config_dst = output_submit_dir / f'{alpha_name}.xml'
        shutil.copy(config_src, config_dst)

        with open(config_dst, 'r') as f:
            content = f.read()
        content = content.replace(identifier, f'{alpha_name}').replace(f'Alpha_XYF_{accounting}', alpha_name)
        with open(config_dst, 'w') as f:
            f.write(content)


def main():
    df = set_the_mapping_between_names()
    copy_all_python_files(df)


if __name__ == '__main__':
    main()
