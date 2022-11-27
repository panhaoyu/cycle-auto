import platform
import shutil

import pandas as pd

from step_1_run_factor_test import base_dir, process_batch
from step_2_generate_excel import get_dataframe, output_best_dir

output_csv_dir = base_dir / 'output-csv'


def main():
    output_csv_dir.mkdir(parents=True, exist_ok=True)
    df: pd.DataFrame = get_dataframe(output_best_dir)
    df.to_csv(output_best_dir / 'data.csv')
    df = df.loc[:, ['accounting', 'operation', 'alpha_sign']]
    shutil.rmtree(output_csv_dir)
    tasks = []
    for accounting, operation, alpha_sign in df.itertuples(index=False):
        tasks.append({
            'accounting': accounting,
            'operation': operation,
            'alpha_sign': alpha_sign,
            'output_dir': output_csv_dir,
            'dump_alpha': 'StatsDumpAlpha',
        })
    if platform.platform() != 'win32':
        process_batch(tasks)


if __name__ == '__main__':
    main()
