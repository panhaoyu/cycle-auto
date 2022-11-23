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


def execute():
    """执行原有的 my_factor_test 脚本，并获取结果"""
    command_results = os.popen(command).readlines()
    selected_line = [l for l in command_results if l.startswith('Alpha_XYF000001')][0]
    result = float(selected_line.split()[2])
    return result


end_marker = "# Auto edit, please don't change"


def change_name(name: str):
    with open(pylib_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    changed_values = {
        'changeable_value': repr(name),
    }
    lines = [l.rstrip('\n') for l in lines]
    for index, line in enumerate(lines):
        if line.endswith(end_marker):
            line = line.split('#')[0].strip()
            name, value = line.split('=')
            name, value = name.strip(), value.strip()
            value = changed_values.get(name, value)
            lines[index] = f'{name} = {value} {end_marker}'
            print(lines[index])
    lines = [f'{l}\n' for l in lines]
    with open(pylib_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def main():
    path = Path('/home/student/work/alpha/pybsim/pylib/variable.xlsx')
    df = pd.read_excel(path, header=None)
    values = list(df.loc[:, 0])
    for i in values:
        change_name(i)
        result = execute()
        print(i, result)


if __name__ == '__main__':
    main()
