# 由于难以更改 config.xml 等一系列的配置文件，故直接再加一层脚本直接调用。
import os
from pathlib import Path

import pandas as pd

command = ('python3 '
           '/home/student/work/alpha/pybsim/bin/my_factor_test.py '
           '/datas/student/AlphaTest/Alpha_XYF000001.pnl.txt')
pylib_file = Path('/home/student/work/alpha/pybsim/pylib/Alpha_XYF000001.py')
output_dir = pylib_file.parent.parent / 'output'
output_dir.mkdir(parents=True, exist_ok=True)


def once(value):
    data = os.popen(command).read()
    print(data)


def main():
    path = Path('/home/student/work/alpha/pybsim/pylib/variable.xlsx')
    df = pd.read_excel(path, header=None)
    values = list(df.loc[:, 0])
    for i in values:
        once(i)
        break


if __name__ == '__main__':
    main()
