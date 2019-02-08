# -*- coding: utf-8 -*-

# Third Party
import pandas as pd

# This Module
from pysankey import sankey
from pysankey.tests.test_fruit import TestFruit


class TestReadmeFruit(TestFruit):
    def test_no_fail_readme(self):
        pd.options.display.max_rows = 8
        sankey(
            [self.data["true"], self.data["predicted"]],
            aspect=20,
            color_dict=self.color_dict,
            fontsize=12,
            figure_name=self.figure_name,
        )
