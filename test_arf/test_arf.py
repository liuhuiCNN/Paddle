import paddle.fluid as fluid
import paddle


paddle.set_device('cpu')

def test_in_static():
    paddle.enable_static()
    # 
    x = fluid.data(name="x", shape=[8, 4, 8, 3, 3], dtype="float32")
    indices = fluid.data(name="indices", shape=[8, 3, 3, 8], dtype="float32")
    out = paddle.vision.ops.arf(x, indices)
    print('out', out) 


def test_in_static_feed():
    paddle.enable_static()
    with fluid.program_guard(fluid.Program()):
        x = fluid.data(name="x", shape=[8, 4, 8, 3, 3], dtype="float32")
        indices = fluid.data(name="indices", shape=[8, 3, 3, 8], dtype="float32")
        #out = fluid.layers.nn.arf(x, indices)
        out = paddle.vision.ops.arf(x, indices)
        out1 = paddle.fluid.layers.cast(out, 'float32') 
        print('out1', out1) 
        #compiled_train_prog = fluid.CompiledProgram(train_prog)
        debug_var = ['cast_0.tmp_0']
        exe = fluid.Executor(fluid.CPUPlace())
        outs = exe.run(fetch_list= debug_var, return_numpy=False)
        print(outs)


def test_in_dygraph():
    paddle.disable_static()
    from paddle.fluid.dygraph import base
    import numpy as np


    out_channels = 8
    in_channels = 4
    nOrientation = 8
    kernel_size = 3
    nRotation = 8
    kH = kernel_size
    kW = kernel_size

    inputs_np = np.random.rand(out_channels, in_channels, nOrientation,
                    kernel_size, kernel_size).astype('float32')

    indidces_np = np.random.randint(1,kernel_size*kernel_size, (nOrientation,
        kH, kW, nRotation))
    
    inputs_np = np.load('input.npy')
    indidces_np = np.load('indices.npy')
    inputs_np = inputs_np.astype(np.float32)
    indidces_np = indidces_np.astype(np.float32)
    inputs_dy = base.to_variable(inputs_np)
    indices_dy = base.to_variable(indidces_np)
    print('inputs_dy', inputs_dy.shape, 'indices_dy', indices_dy.shape)
    print('call ops')
    out = paddle.vision.ops.arf(inputs_dy, indices_dy)
    #out = paddle.fluid.layers.nn.arf(inputs_dy, indices_dy)
    print('out', out.numpy().shape)
    expect_out = np.load('arf_out.npy')
    print(np.sum(expect_out - out.numpy()))


#test_in_static()
test_in_dygraph()

