# -*- coding: utf-8 -*-

# This Module
from pysankey import sankey
from pysankey.tests.test_customer_goods import TestCustomerGood


class TestReadmeCustomerGood(TestCustomerGood):
    def test_no_fail_readme(self):
        # This is not working yet...
        sankey(
            [self.data["customer"], self.data["good"]],
            weights=[None, self.data["revenue"]],
            aspect=20,
            fontsize=20,
            figure_name=self.figure_name,
        )
