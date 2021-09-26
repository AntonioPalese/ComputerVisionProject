from YOOXTester2 import YOOXTester
import platform
from ExtractParsed import make
import os


def make_test(recall, category):

    if platform.system() == 'Windows':
        root = "D:\\datasetCV_nomanichini_classificato"
        imgs_path = os.path.join(root, category)
        weights_path = "data\\model-weights-lower_body.pth"

    elif platform.system() == 'Linux' or platform.system() == 'Darwin':
        root = '../DatasetFolder/DatasetCV'
        imgs_path = os.path.join(root, category)
        weights_path = "./weights-pytorch-upper_body.pth"

    tester = YOOXTester(root, imgs_path, category, recall, platform.system(), weights_path)
    tester(verbose=False)


def main():
    part = 'Pants'  # Upper-clothes o Pants
    category = 'lower_body'  # upper_body o lower_body
    Fields = 'bermudas'  # upper o bermudas; fields for the preprocessor
    make(category=category, Fields=Fields, part=part)
    make_test(10, category)


if __name__ == '__main__':
    main()