import shutil

import pandas as pd

from step_1_run_factor_test import base_dir
from step_2_generate_excel_and_modify_sign import get_dataframe, output_best_dir
from step_3_generate_csv import output_csv_dir
from step_4_generate_images import iter_identifiers

output_submit_dir = base_dir / 'output-submit'


def set_the_mapping_between_names():
    df: pd.DataFrame = get_dataframe(output_best_dir)
    df = df.sort_values('accounting').reset_index(drop=True)
    df['alpha_name'] = [f'Alpha_XYF_{i + 101}' for i in df.index]
    # noinspection PyTypeChecker
    df.to_csv(output_submit_dir / 'data.csv', index=None)
    return df


def copy_files(df: pd.DataFrame):
    for identifier, data in iter_identifiers(df):
        src_dir = output_best_dir / identifier
        # 复制python文件
        python_files = list((src_dir).glob('*.py'))
        python_files = [p for p in python_files if not p.stem == 'Alpha']
        python_src = python_files[0]
        alpha_name = data['alpha_name']
        accounting = data['accounting']
        python_dst = output_submit_dir / f'{alpha_name}.py'
        shutil.copy(python_src, python_dst)

        # 复制config文件并修改
        config_src = src_dir / 'config.xml'
        config_dst = output_submit_dir / f'{alpha_name}.xml'
        shutil.copy(config_src, config_dst)
        with open(config_dst, 'r') as f:
            content = f.read()
        content = content.replace(identifier, f'{alpha_name}').replace(f'Alpha_XYF_{accounting}', alpha_name)
        with open(config_dst, 'w') as f:
            f.write(content)

        # 复制 step2-stdout.txt 文件
        stdout_src = next(src_dir.glob('*-step2-stdout.txt'))
        stdout_dst = output_submit_dir / f'{alpha_name}.txt'
        shutil.copy(stdout_src, stdout_dst)

        # 复制 png 文件
        part_of_identifier = identifier.removesuffix('StatsBacktest')
        png_src = next(output_csv_dir.glob(f'{part_of_identifier}*/Alpha_*.png'))
        png_dst = output_submit_dir / f'{alpha_name}.png'
        shutil.copy(png_src, png_dst)


def main():
    shutil.rmtree(output_submit_dir, ignore_errors=True), output_submit_dir.mkdir()
    df = set_the_mapping_between_names()
    copy_files(df)


if __name__ == '__main__':
    main()
