# 由于难以更改 config.xml 等一系列的配置文件，故直接再加一层脚本直接调用。
import multiprocessing
import os
import shutil
from pathlib import Path

import pandas as pd
from lxml import etree

base_dir = Path(__file__).parent
data_dir = Path('/datas/student/AlphaTest')
bin_template_dir = base_dir / 'bin.tpl'
pylib_template_file = base_dir / 'Alpha_XYF000001.tpl.py'
excel_file = base_dir / 'variable.xlsx'


def process(accounting: str, operation: str):
    bin_dir = base_dir / f'bin-{accounting}-{operation}'
    pylib_file = bin_dir / f'Alpha_XYF_{accounting}.py'
    pysim_file = bin_dir / 'pybsim'
    config_file = bin_dir / 'config.xml'
    my_factor_test_file = bin_dir / 'my_factor_test.py'
    pnl_file = data_dir / f'{pylib_file.stem}.pnl.txt'
    output_dir = base_dir / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)

    def copy_bin():
        shutil.rmtree(bin_dir, ignore_errors=True)
        shutil.copytree(bin_template_dir, bin_dir)
        shutil.copy(pylib_template_file, pylib_file)
        pysim_file.chmod(0o755)

    def set_changeable_values():
        with open(pylib_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        changed_values = {
            'changeable_value': repr(accounting),
        }
        lines = [l.rstrip('\n') for l in lines]
        end_marker = "# Auto edit, please don't change"
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

    def set_config():
        with open(config_file, 'rb') as f:
            config_content = f.read()
        tree = etree.fromstring(config_content)

        # 设置 alphacode 的路径
        element = tree.xpath('//Portfolio')[0]
        element.attrib['alphacode'] = str(bin_dir)
        element = element.xpath('./Alpha')[0]
        element.attrib['id'] = pylib_file.stem
        element = element.xpath('./Config')[0]
        element.attrib['alphaname'] = pylib_file.stem
        config_content = etree.tostring(tree)
        with open(config_file, 'wb') as f:
            f.write(config_content)

    def execute():
        """执行原有的 my_factor_test 脚本，并获取结果"""
        print('Executing ...')
        os.chdir(bin_dir)
        os.popen(f'PYTHONPATH=$PYTHONPATH:{bin_dir} ./pybsim').read()
        print('Executed.')
        command_results = os.popen(f'python3 {my_factor_test_file} {pnl_file}').readlines()
        selected_line = [l for l in command_results if l.startswith(pylib_file.stem)][0]
        return float(selected_line.split()[2])

    try:
        copy_bin()
        set_changeable_values()
        set_config()
        if (result := execute()) > 0.2:
            print(f'Find available: {accounting} -> {result}, copy to result dir.')
            shutil.copy(pylib_file, output_dir / f'{accounting}.py')
    finally:
        shutil.rmtree(bin_dir)


def main():
    df = pd.read_excel(excel_file, header=None)
    values = list(df.loc[:, 0])
    with multiprocessing.Pool(10) as pool:
        pool.map(process, values)


if __name__ == '__main__':
    process('CASH_RECP_SG_AND_RS')
    # main()
