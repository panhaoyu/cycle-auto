import os
import click

@click.command()
@click.argument('filepath', required=True)
def run(filepath):
    simsum_cmd = f'python3 /datas/share/newAccount/work/alpha/Alpha4/bin/tools/simsum2.py {filepath}'
    print()
    os.system(simsum_cmd)
    corr_cmd = f'python3 /datas/share/AlphaPool/checkcorr2.py -c {filepath}'
    print()
    os.system(corr_cmd)

if __name__ == '__main__':
    run()