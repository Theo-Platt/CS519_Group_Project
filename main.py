import argparse
from func_codes.converter import Converter
import cv2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='main',
        description='Convert Math pictures to Latex',
        epilog='MathPicToLatex')

    parser.add_argument('-p', metavar='--path', help="Path to the image")

    args = parser.parse_args()
    # get the path
    path = args.p

    # read the image
    src = cv2.imread(path)
    if src is None:
        print(f'Could not open or find the image at path {path}.')
        exit(0)

    # convert the img to number
    converter = Converter()
    print(converter.convert_img_to_latex(src))

    # cv2.waitKey(0)
    # cv2.destroyAllWindows() 