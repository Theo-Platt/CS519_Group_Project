from train_codes import single_gen, single_symbol_recognizer
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    prog='python3 tools.py',
    description='Tools to generate dataset, train/save models, and evaluate models',
    epilog='Toolsets')

    parser.add_argument('--option', metavar='--path', help="Path to the image")

    args = parser.parse_args()
    # generate data
    single_gen.main()
