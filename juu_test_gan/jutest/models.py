from jutest.prepared import Prepared
from jutest.train import Train
import torch


class JuTest:
    def __init__(self):
        Prepared().process()
        # Train().process()


if __name__ == '__main__':
    JuTest()
