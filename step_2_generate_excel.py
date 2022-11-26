import json
from pathlib import Path

import pandas as pd

from step_1_run_factor_test import process_batch

base_dir = Path(__file__).parent
output_dir = base_dir / 'output'


def get_dataframe() -> pd.DataFrame:
    data = []
    keys = 'accounting operation sharpe'.split()
    for i in output_dir.glob('*/meta.json'):
        with open(i, 'r', encoding='utf-8') as f:
            datum = json.load(f)
        data.append(tuple(datum[i] for i in keys))
    df = pd.DataFrame(data, columns=keys)
    df.loc[df['operation'].apply(lambda j: j is None), 'operation'] = 'Empty'
    return df


def main():
    df = get_dataframe()
    pt = df.pivot_table('sharpe', index=['accounting'], columns=['operation'])
    pt.to_excel(output_dir / 'sharpe.xlsx')
    available = []
    for accounting, operations in pt.iterrows():
        max_value = max(abs(operations))
        selected = [(k, v) for k, v in operations.to_dict().items() if abs(v) == max_value and abs(v) > 0.75]
        if not selected:
            continue
        operation, sharpe = selected[0]
        available.append((
            accounting,
            operation,
            1 if sharpe > 0 else -1,  # alpha sign
            base_dir / 'output-best',  # output dir
        ))
    process_batch(available)


if __name__ == '__main__':
    main()
