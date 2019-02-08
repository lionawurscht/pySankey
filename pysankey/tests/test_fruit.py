# -*- coding: utf-8 -*-

# Third Party
import pandas as pd

# This Module
from pysankey.tests.generic_test import GenericTest


class TestFruit(GenericTest):

    """ Permit to test sankey with the data in fruit.txt """

    def setUp(self):
        self.figure_name = "fruit"
        self.data = pd.read_csv(
            "pysankey/fruits.txt", sep=" ", names=["true", "predicted"]
        )
        self.color_dict = {
            "apple": "#f71b1b",
            "blueberry": "#1b7ef7",
            "banana": "#f3f71b",
            "lime": "#12e23f",
            "orange": "#f78c1b",
        }
