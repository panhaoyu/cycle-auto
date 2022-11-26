import json
import platform
import shutil
from pathlib import Path

import pandas as pd

from step_1_run_factor_test import process_batch

base_dir = Path(__file__).parent
output_dir = base_dir / 'output'


def get_dataframe() -> pd.DataFrame:
    data = []
    keys = 'accounting operation'.split()
    for i in output_dir.glob('*/meta.json'):
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
    df = pd.DataFrame(data, columns=[*keys, 'sharpe'])
    df.loc[df['operation'].apply(lambda j: j is None), 'operation'] = 'Empty'
    return df


def main():
    df = get_dataframe()
    df.to_excel(output_dir / 'data.xlsx')
    pt = df.pivot_table('sharpe', index=['accounting'], columns=['operation'])
    pt.to_excel(output_dir / 'sharpe.xlsx')
    available = []
    new_output_dir = base_dir / 'output-best'
    shutil.rmtree(new_output_dir)
    for accounting, operations in pt.iterrows():
        max_value = max(abs(operations))
        selected = [(k, v) for k, v in operations.to_dict().items() if abs(v) == max_value and abs(v) > 0.75]
        if not selected:
            continue
        operation, sharpe = selected[0]
        available.append((
            accounting,
            None if operation == 'Empty' else operation,
            1 if sharpe > 0 else -1,  # alpha sign
            new_output_dir,  # output dir
        ))
    if platform.platform() != 'win32':
        process_batch(available)


if __name__ == '__main__':
    main()
