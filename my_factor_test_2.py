# 由于难以更改 config.xml 等一系列的配置文件，故直接再加一层脚本直接调用。
import datetime
import os
import shutil
from pathlib import Path

import pandas as pd
from lxml import etree

base_dir = Path(__file__).parent
data_dir = Path('/datas/student/AlphaTest')

bin_template_dir = base_dir / 'bin.tpl'
bin_dir = base_dir / 'bin-3'

pylib_template_file = base_dir / 'Alpha_XYF000001.tpl.py'
pylib_file = bin_dir / 'Alpha_XYF000004.py'

pysim_file = bin_dir / 'pybsim'
config_file = bin_dir / 'config.xml'
my_factor_test_file = bin_dir / 'my_factor_test.py'
excel_file = base_dir / 'variable.xlsx'
pnl_file = data_dir / f'{pylib_file.stem}.pnl.txt'
output_dir = base_dir / 'output'
output_dir.mkdir(parents=True, exist_ok=True)


def execute():
    """执行原有的 my_factor_test 脚本，并获取结果"""
    print('Executing ...')
    os.chdir(bin_dir)
    os.popen(f'PYTHONPATH=$PYTHONPATH:{bin_dir} ./pybsim').read()
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


def all_for_one_value(value):
    shutil.rmtree(bin_dir, ignore_errors=True)
    shutil.copytree(bin_template_dir, bin_dir)
    shutil.copy(pylib_template_file, pylib_file)
    pysim_file.chmod(0o755)
    change_name(value)
    with open(config_file, 'rb') as f:
        config_content = f.read()
    tree = etree.fromstring(config_content)
    element = tree.xpath('//Portfolio')[0]
    element.attrib['alphacode'] = str(bin_dir)
    element = element.xpath('./Alpha')[0]
    element.attrib['id'] = pylib_file.stem
    element = element.xpath('./Config')[0]
    element.attrib['alphaname'] = pylib_file.stem
    config_content = etree.tostring(tree)
    with open(config_file, 'wb') as f:
        f.write(config_content)
    result = execute()
    if result > 0.2:
        print(f'Find available: {value} -> {result}, copy to result dir.')
        shutil.copy(pylib_file, output_dir / datetime.datetime.now().strftime('%Y%m%d-%H%M%S.py'))


def main():
    df = pd.read_excel(excel_file, header=None)
    values = list(df.loc[:, 0])
    for i in values:
        all_for_one_value(i)
        break


if __name__ == '__main__':
    main()
