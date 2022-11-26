import json
from pathlib import Path

import pandas as pd

output_dir = Path(__file__).parent / 'output'


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
    df['sharpe'] = abs(df['sharpe'])
    pt = df.pivot_table('sharpe', index=['accounting'], columns=['operation'])
    pt.to_excel(output_dir / 'sharpe.xlsx')
    available = []
    for accounting, operations in pt.iterrows():
        max_value = max(operations)
        selected = [k for k, v in operations.to_dict().items() if v == max_value and v > 0.75]
        if selected:
            available.append((accounting, selected[0]))
    for i in available:
        print(i)


if __name__ == '__main__':
    main()
