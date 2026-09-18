!pip install pycuda

#load engine back into the memory
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(TRT_LOGGER)

with open('yolov8n.engine', 'rb') as f:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

print("Engine loaded successfully")


#Inspect the engine's input/output tensor names
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    shape = engine.get_tensor_shape(name)
    dtype = engine.get_tensor_dtype(name)
    mode = engine.get_tensor_mode(name)
    print(f"Tensor {i}: name='{name}', shape={shape}, dtype={dtype}, mode={mode}")

