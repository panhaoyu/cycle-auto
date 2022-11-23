# 由于难以更改 config.xml 等一系列的配置文件，故直接再加一层脚本直接调用。
import datetime
import os
import shutil
from pathlib import Path

import pandas as pd

base_dir = Path(__file__).parent
data_dir = Path('/datas/student/AlphaTest')
bin_dir = base_dir / 'bin'
pylib_file = bin_dir / 'Alpha_XYF000001.py'
pysim_file = bin_dir / 'pybsim'
my_factor_test_file = bin_dir / 'my_factor_test.py'
excel_file = base_dir / 'variable.xlsx'
pnl_file = data_dir / f'{pylib_file.stem}.pnl.txt'
pysim_file.chmod(0o755)
output_dir = base_dir / 'output'
output_dir.mkdir(parents=True, exist_ok=True)


def execute():
    """执行原有的 my_factor_test 脚本，并获取结果"""
    print('Executing ...')
    os.chdir(bin_dir)
    os.popen(f'PYTHONPATH=$PYTHONPATH:{bin_dir} {pysim_file}').read()
    print('Executed.')
    command_results = os.popen(f'python3 {my_factor_test_file} {pnl_file}').readlines()
    selected_line = [l for l in command_results if l.startswith(pylib_file.stem)][0]
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
    lines = [f'{l}\n' for l in lines]
    with open(pylib_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def main():
    df = pd.read_excel(excel_file, header=None)
    values = list(df.loc[:, 0])
    for i in values:
        change_name(i)
        result = execute()
        if result > 0.2:
            print(f'Find available: {i} -> {result}, copy to result dir.')
            shutil.copy(pylib_file, output_dir / datetime.datetime.now().strftime('%Y%m%d-%H%M%S.py'))
        break


if __name__ == '__main__':
    main()
