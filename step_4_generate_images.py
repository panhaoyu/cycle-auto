import multiprocessing
from typing import List, Tuple, Any, Dict

import pandas as pd

from step_1_run_factor_test import get_identifier, global_dump_alpha_dir, run_command
from step_2_generate_excel_and_modify_sign import get_dataframe, output_best_dir
from step_3_generate_csv import output_csv_dir


def iter_identifiers(df: pd.DataFrame) -> List[Tuple[str, Dict[str, Any]]]:
    identifier_keys = ['accounting', 'operation', 'alpha_sign', 'dump_alpha']
    for k, v in df.to_dict('index').items():
        identifier = get_identifier(*(v[k] for k in identifier_keys))
        yield identifier, v


def main():
    df: pd.DataFrame = get_dataframe(output_csv_dir)
    df.to_csv(output_best_dir / 'data.csv')
    df = df.loc[:, ['accounting', 'operation', 'alpha_sign', 'dump_alpha']]
    data = []
    for accounting, operation, alpha_sign, dump_alpha in df.itertuples(index=False):
        identifier = get_identifier(accounting, operation, alpha_sign, dump_alpha)
        print(f'Generating image for {identifier}')
        output_dir = output_csv_dir / identifier
        data.append(([
                         'python',
                         '/home/student/work/alpha/pybsim/bin.tpl/backtest_new.py',
                         str(global_dump_alpha_dir / identifier / 'dumplAlphas.csv'),
                         str(output_dir),
                     ], 'step-4-generate-images', output_dir))
    with multiprocessing.Pool(4) as pool:
        pool.starmap(run_command, data)


if __name__ == '__main__':
    main()
