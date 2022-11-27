from step_2_generate_excel import get_dataframe, output_best_dir


def main():
    df = get_dataframe(output_best_dir)
    df.to_csv(output_best_dir / 'data.csv')
    print(df)


if __name__ == '__main__':
    main()
