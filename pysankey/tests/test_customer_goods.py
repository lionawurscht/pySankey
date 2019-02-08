# -*- coding: utf-8 -*-

# Third Party
import pandas as pd

# This Module
from pysankey.tests.generic_test import GenericTest


class TestCustomerGood(GenericTest):

    """ Permit to test sankey with the data in customers-goods.csv """

    def setUp(self):
        self.figure_name = "customer-good"
        self.data = pd.read_csv(
            "pysankey/customers-goods.csv",
            sep=",",
            names=["id", "customer", "good", "revenue"],
            header=0,
        )
