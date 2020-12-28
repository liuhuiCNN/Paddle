#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import unittest
import numpy as np

import paddle.fluid.core as core
import paddle.fluid as fluid
from op_test import OpTest, skip_check_grad_ci
from paddle.fluid import Program, program_guard

import paddle
paddle.enable_static()
paddle.set_device('cpu')


@skip_check_grad_ci(reason="the OP test do not support arf OP!")
class TestARFOp(OpTest):
    def setUp(self):
        self.__class__.op_type="arf"
        self.op_type = "arf"
        self.use_cudnn = False
        self.use_cuda = False
        self.use_mkldnn = False
        self.data_format = "AnyLayout"
        self.dtype = np.float64
        self.no_need_check_grad = True


        self.input_weight = np.load('/paddle/input.npy').astype(np.float64)
        self.indices = np.load('/paddle/indices.npy').astype(np.float64)
        self.input_weight = self.input_weight.astype(np.float64)
        self.indices = self.indices.astype(np.float64)
        arf_result = np.load('/paddle/arf_out.npy')
        arf_result = arf_result.astype(self.dtype)

        self.inputs = {
            'InputWeight': OpTest.np_dtype_to_fluid_dtype(self.input_weight),
            'Indices': OpTest.np_dtype_to_fluid_dtype(self.indices)
        }
        self.attrs = {
            "use_cudnn": self.use_cudnn,
            "use_mkldnn": self.use_mkldnn,
            "mkldnn_data_type": "float32",
        }
        self.outputs = {'Output': arf_result}


    def dygraph_check(self):
        paddle.set_device('cpu')
        paddle.disable_static(self.place)
        np_x = np.load('/paddle/input.npy')
        np_x = np_x.astype(np.float64)
        x = paddle.to_tensor(np_x)
        np_indices = np.load('/paddle/indices.npy')
        np_indices = np_indices.astype(np.float64)
        indices = paddle.to_tensor(np_indices)
        out = paddle.vision.ops.arf(x, indices)
        out_ref = np.load('/paddle/arf_out.npy')
        self.assertEqual(np.allclose(out_ref, out.numpy()), True)
        paddle.enable_static()


if __name__ == '__main__':
    unittest.main()

