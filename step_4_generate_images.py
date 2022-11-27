# /home/student/work/alpha/pybsim/bin/backtest_new.py
# /datas/student/AlphaTest/dumplAlphas.csv
# /home/student/work/alpha/pybsim/output


import pandas as pd

from step_2_generate_excel import get_dataframe, output_best_dir
from step_3_generate_csv import output_csv_dir


def main():
    df: pd.DataFrame = get_dataframe(output_csv_dir)
    df.to_csv(output_best_dir / 'data.csv')
    # df = df.loc[:, ['accounting', 'operation', 'alpha_sign']]
    # shutil.rmtree(output_csv_dir)
    # tasks = []
    # for accounting, operation, alpha_sign in df.itertuples(index=False):
    #     tasks.append({
    #         'accounting': accounting,
    #         'operation': operation,
    #         'alpha_sign': alpha_sign,
    #         'output_dir': output_csv_dir,
    #         'dump_alpha': 'StatsDumpAlpha',
    #     })
    # if platform.platform() != 'win32':
    #     process_batch(tasks)


if __name__ == '__main__':
    main()
