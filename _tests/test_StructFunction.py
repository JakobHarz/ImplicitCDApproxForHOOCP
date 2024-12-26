import unittest
from Core.tools import StructFunction
from casadi.tools import struct_symSX, entry
import casadi as ca

class TestStructFunction(unittest.TestCase):

    def test_singleCall(self):
        testStruct = struct_symSX([
            entry('a'),
            entry('b')
        ])

        input = ca.SX.sym('input', 1)
        output = testStruct(ca.vertcat(input, 2 * input))

        test_f = StructFunction('test', [input], [output], struct = testStruct)
        result = test_f(1)

        assert result['a'] == 1
        assert result['b'] == 2

    def test_map(self):

        testStruct = struct_symSX([
            entry('a'),
            entry('b')
        ])

        input = ca.SX.sym('input', 1)
        output = testStruct(ca.vertcat(input, 2 * input))

        test_f = StructFunction('test', [input], [output], struct = testStruct)
        result = test_f.map(4)([1,2,3,4])

        assert result[0,'a'] == 1
        assert result[0,'b'] == 2
        assert result[1,'a'] == 2