# 由于难以更改 config.xml 等一系列的配置文件，故直接再加一层脚本直接调用。
import json
import multiprocessing
import os
import shutil
from pathlib import Path
from subprocess import Popen, PIPE
from typing import Union, List, Tuple

import pandas as pd
from lxml import etree

base_dir = Path(__file__).parent
global_dump_alpha_dir = Path('/datas/student/AlphaTest-Auto')
bin_template_dir = base_dir / 'bin.tpl'
pylib_template_file = base_dir / 'Alpha_XYF000001.tpl.py'
excel_file = base_dir / 'variable.xlsx'
global_temp_dir = base_dir / 'temp'
global_output_dir = base_dir / 'output'
lock = multiprocessing.Lock()


def process(
        accounting: str,
        operation: Union[str, None],
        alpha_sign: float = 1,
        output_dir=global_output_dir,
        dump_alpha='StatsBacktest',  # 在后面生成csv的时候要改成StatsDumpAlpha
):
    operation = None if operation == 'Empty' else operation
    # 任何情况下都要确保这个标识符不重复，即同样的标识符一定对应着完全一样的项目
    identifier = '-'.join((f'{accounting}',
                           f'{"p" if alpha_sign > 0 else "n"}{abs(alpha_sign):.0f}',
                           f'{"Empty" if operation is None else operation}',
                           dump_alpha))
    bin_dir = global_temp_dir / f'{identifier}'
    pylib_file = bin_dir / f'Alpha_XYF_{accounting}.py'
    pysim_file = bin_dir / 'pybsim'
    config_file = bin_dir / 'config.xml'
    my_factor_test_file = bin_dir / 'my_factor_test.py'
    (dump_alpha_dir := global_dump_alpha_dir / identifier).mkdir(parents=True, exist_ok=True)
    pnl_file = dump_alpha_dir / f'{pylib_file.stem}.pnl.txt'
    dump_alphas_csv_file = dump_alpha_dir / 'dumplAlphas.csv'
    (output_dir := output_dir / identifier).mkdir(parents=True, exist_ok=True)

    def copy_bin():
        shutil.rmtree(bin_dir, ignore_errors=True)
        shutil.copytree(bin_template_dir, bin_dir)
        shutil.copy(pylib_template_file, pylib_file)
        pysim_file.chmod(0o755)

    def set_changeable_values():
        with open(pylib_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        changed_values = {
            'variable_accounting': repr(accounting),
            'variable_alpha_sign': repr(alpha_sign),
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
        shutil.copy(pylib_file, output_dir / 'Alpha.py')

    def set_config():
        with open(config_file, 'rb') as f:
            config_content = f.read()
        tree = etree.fromstring(config_content)

        # 设置 alphacode 的路径
        portfolio_element = tree.xpath('//Portfolio')[0]
        portfolio_element.attrib['alphacode'] = str(bin_dir)
        alpha_element = portfolio_element.xpath('./Alpha')[0]
        alpha_element.attrib['id'] = pylib_file.stem
        config_element = alpha_element.xpath('./Config')[0]
        config_element.attrib['alphaname'] = pylib_file.stem

        # 设置 operation
        operations_element = alpha_element.xpath('./Operations')[0]
        operation_element = operations_element.xpath('./Operation')[0]
        if operation is None:
            operations_element.remove(operation_element)
        else:
            operation_element.attrib['moduleId'] = operation

        # 设置 dump alpha
        stats_element = portfolio_element.xpath('./Stats')[0]
        stats_element.attrib['moduleId'] = dump_alpha

        # 设置 dump alpha directory
        tree.xpath('//Modules/Module[@id="StatsDumpAlpha"]')[0].attrib['dumpAlphaDir'] = str(dump_alpha_dir)
        tree.xpath('//Modules/Module[@id="StatsBacktest"]')[0].attrib['dumpAlphaDir'] = str(dump_alpha_dir)
        alpha_element.attrib['dumpAlphaDir'] = str(dump_alpha_dir)

        config_content = etree.tostring(tree)
        with open(config_file, 'wb') as f:
            f.write(config_content)
        shutil.copy(config_file, output_dir / 'config.xml')

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
        with open(output_dir / f'{name}-stdout.txt', 'w', encoding='utf-8') as f:
            f.write(stdout)
        with open(output_dir / f'{name}-stderr.txt', 'w', encoding='utf-8') as f:
            f.write(stderr)
        return stdout, stderr

    def execute():
        """执行原有的 my_factor_test 脚本，并获取结果"""
        print(f'Executing {identifier} ...')
        os.chdir(bin_dir)
        os.environ['PYTHONPATH'] = os.environ.get('PYTHONPATH', '') + f':{bin_dir}'
        run(f'./pybsim', f'{identifier}-step1')
        run(['python3', my_factor_test_file, pnl_file], f'{identifier}-step2')

    def save_result():
        data = {
            'accounting': accounting,
            'operation': operation,
            'alpha-sign': alpha_sign,
            'dump-alpha': dump_alpha,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(pylib_file, output_dir / f'{accounting}.py')
        shutil.copy(config_file, output_dir / 'config.xml')
        # 由于这个文件过大，不去拷贝它
        # dump_alphas_csv_file.exists() and shutil.copy(dump_alphas_csv_file, output_dir / 'dumplAlpha.csv')
        with open(output_dir / 'meta.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    try:
        copy_bin()
        set_changeable_values()
        set_config()
        execute()
        save_result()
    except:
        pass
    finally:
        shutil.rmtree(bin_dir, ignore_errors=True)


def process_wrapper(values):
    """参数一点点变多了，使用此函数进行包装，可以实现多进程版本的kwargs参数字典"""
    process(**values)


def process_batch(values):
    with multiprocessing.Pool(64) as pool:
        # noinspection PyTypeChecker
        pool.map(process_wrapper, values)


def main():
    df = pd.read_excel(excel_file, header=None)
    accounting_list = list(df.loc[:, 0])
    operations: List[Tuple[str, Union[str, None]]]
    operations = [*'AlphaOpIndNeut AlphaOpMktCapNeut AlphaOpCapSecNeut'.split(), None]
    # operations = [*'AlphaOpIndNeut AlphaOpIndNeut_new AlphaOpMktCapNeut AlphaOpCapSecNeut'.split(), None]
    values = [{'accounting': accounting, 'operation': operation}
              for accounting in accounting_list for operation in operations]
    process_batch(values)


if __name__ == '__main__':
    # process('CASH_RECP_SG_AND_RS', None)
    try:
        global_temp_dir.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(global_output_dir, ignore_errors=True)
        main()
    finally:
        shutil.rmtree(global_temp_dir, ignore_errors=True)
