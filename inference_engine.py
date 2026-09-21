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



#preprocessing
import cv2
import numpy as np

img = cv2.imread('test.jpg')
print("Original shape:", img.shape)

img_resized = cv2.resize(img, (640, 640))

img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

img_normalized = img_rgb.astype(np.float32) / 255.0

img_chw = np.transpose(img_normalized, (2, 0, 1))

input_data = np.expand_dims(img_chw, axis=0)

input_data = np.ascontiguousarray(input_data)

print("Final input shape:", input_data.shape)
print("Data type:", input_data.dtype)



#allocating GPU memory
output_shape = (1, 84, 8400)
output_data = np.empty(output_shape, dtype=np.float32)

d_input = cuda.mem_alloc(input_data.nbytes)
d_output = cuda.mem_alloc(output_data.nbytes)

cuda.memcpy_htod(d_input, input_data)

print("Input allocated:", d_input)
print("Output allocated:", d_output)
print("Input copied to GPU")


#bind memory addresses and run the engine
context.set_tensor_address("images", int(d_input))
context.set_tensor_address("output0", int(d_output))

stream = cuda.Stream()

context.execute_async_v3(stream_handle=stream.handle)

stream.synchronize()

print("Inference executed")



#copy results back to cpu
cuda.memcpy_dtoh(output_data, d_output)

print("Output copied back to CPU")
print("Output shape:", output_data.shape)
print("Sample values (first detection candidate):")
print(output_data[0, :, 0])
