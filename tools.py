from train_codes import single_gen, single_symbol_recognizer
from eval_codes  import evaluate
import argparse

selection = {
    "generate": single_gen.main,
    "train":    single_symbol_recognizer.main,
    "evaluate": evaluate.main,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='python3 tools.py',
        description='Tools to generate dataset, train/save models, and evaluate models',
        epilog='Toolsets'
        )

    parser.add_argument(
        "option", 
        choices=['generate','train','evaluate'], 
        type=str, 
        help="Decide whether dataset is generated, models are trained/saved, or models are evaluated."
        )

    args = parser.parse_args()

    run = selection.get(args.option, "")
    run()
