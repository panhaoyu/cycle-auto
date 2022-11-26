# 由于难以更改 config.xml 等一系列的配置文件，故直接再加一层脚本直接调用。
import multiprocessing
import os
import shutil
from pathlib import Path
from subprocess import Popen, PIPE
from typing import Union, List, Tuple

import pandas as pd
from lxml import etree

base_dir = Path(__file__).parent
data_dir = Path('/datas/student/AlphaTest')
bin_template_dir = base_dir / 'bin.tpl'
pylib_template_file = base_dir / 'Alpha_XYF000001.tpl.py'
excel_file = base_dir / 'variable.xlsx'

lock = multiprocessing.Lock()


def process(accounting: str, operation: Union[str, None]):
    identifier = f'{accounting}' if operation is None else f'{accounting}-{operation}'
    bin_dir = base_dir / f'bin-{identifier}'
    pylib_file = bin_dir / f'Alpha_XYF_{accounting}.py'
    pysim_file = bin_dir / 'pybsim'
    config_file = bin_dir / 'config.xml'
    my_factor_test_file = bin_dir / 'my_factor_test.py'
    pnl_file = data_dir / f'{pylib_file.stem}.pnl.txt'

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

    def run(args: Union[str, List[str]], name: str):
        popen = Popen(args, stdout=PIPE, stderr=PIPE)
        stdout, stderr = popen.communicate()
        stdout, stderr = stdout.decode(), stderr.decode()
        popen.wait()
        with lock:
            print('-' * 80), print(name)
            stdout and (print('STDOUT:'), print(stdout))
            stderr and (print('STDERR:'), print(stderr))
            print('-' * 80)
        return stdout, stderr

    def execute():
        """执行原有的 my_factor_test 脚本，并获取结果"""
        print(f'Executing {identifier} ...')
        os.chdir(bin_dir)
        os.environ['PYTHONPATH'] = os.environ.get('PYTHONPATH', '') + f':{bin_dir}'
        run(f'./pybsim', f'Executing result of {identifier}')
        stdout, stderr = run(['python3', my_factor_test_file, pnl_file], f'Calculating result of {identifier}')
        selected_line = [l for l in stdout.splitlines() if l.startswith(pylib_file.stem)][0]
        return float(selected_line.split()[2])

    def save_result():
        print(f'Find available: {accounting} -> {result}, copy to result dir.')
        output_dir = base_dir / 'output' / identifier
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(pylib_file, output_dir / f'{accounting}.py')
        shutil.copy(config_file, output_dir / 'config.xml')

    try:
        copy_bin()
        set_changeable_values()
        set_config()
        if abs(result := execute()) > 0.2:
            save_result()
    finally:
        shutil.rmtree(bin_dir)


def main():
    df = pd.read_excel(excel_file, header=None)
    accounting_list = list(df.loc[:, 0])
    operations: List[Tuple[str, Union[str, None]]]
    operations = [*'AlphaOpIndNeut AlphaOpIndNeut_new AlphaOpMktCapNeut AlphaOpCapSecNeut'.split(), None]
    values = [(accounting, operation) for accounting in accounting_list for operation in operations]
    with multiprocessing.Pool(64) as pool:
        # noinspection PyTypeChecker
        pool.starmap(process, values)


if __name__ == '__main__':
    # process('CASH_RECP_SG_AND_RS', None)
    main()
