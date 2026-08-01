import unittest
import pandas as pd
import MD

class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()


test = pd.DataFrame({'A': (1,2,3),
                     'B' : (4,1,6)})

m_space = get_unit_space(test)
'''
print(m_space.mean)
print(m_space.std)
print(m_space.corr_matrix_inv)
print(m_space.z)
'''
md_vec = get_md(m_space)
print(md_vec)