import json
import platform
import shutil
from pathlib import Path

import pandas as pd

from step_1_run_factor_test import process_batch, base_dir, global_output_dir as output_dir

output_best_dir = base_dir / 'output-best'


def get_dataframe(directory: Path = output_dir) -> pd.DataFrame:
    """将一个文件夹下的所有结果整理为 DataFrame ，以方便进行筛选和分析。

    Args:
        directory: output 或者 output-best 等输入结果的保存文件夹。

    Returns:
        从 meta.json 中读取得到的结果。
    """
    data = []
    keys = 'accounting operation alpha-sign dump-alpha'.split()
    for i in directory.glob('*/meta.json'):
        with open(i, 'r', encoding='utf-8') as f:
            datum = json.load(f)
        with open(next(i.parent.glob('*step2-stdout.txt')), 'r', encoding='utf-8') as f:
            stdout = f.readlines()
        accounting = datum['accounting']
        selected_lines = [l for l in stdout if l.startswith(f'Alpha_XYF_{accounting}')]
        if not selected_lines:
            print(f'Process failed: {i.parent.stem}')
            continue
        sharpe = float(selected_lines[0].split()[2])
        data.append((*(datum[i] for i in keys), sharpe))
    df = pd.DataFrame(data, columns=[*(k.lower().replace('-', '_') for k in keys), 'sharpe'])
    df.loc[df['operation'].apply(lambda j: j is None), 'operation'] = 'Empty'
    return df


def main():
    df = get_dataframe()
    df.to_excel(output_dir / 'data.xlsx')
    pt = df.pivot_table('sharpe', index=['accounting'], columns=['operation'])
    pt.to_excel(output_dir / 'sharpe.xlsx')
    available = []
    shutil.rmtree(output_best_dir)
    for accounting, operations in pt.iterrows():
        max_value = max(abs(operations))
        selected = [(k, v) for k, v in operations.to_dict().items() if abs(v) == max_value and abs(v) > 0.75]
        if not selected:
            continue
        operation, sharpe = selected[0]
        available.append({
            'accounting': accounting,
            'operation': None if operation == 'Empty' else operation,
            'alpha_sign': 1 if sharpe > 0 else -1,
            'output_dir': output_best_dir,
        })
    if platform.platform() != 'win32':
        process_batch(available)


if __name__ == '__main__':
    main()
