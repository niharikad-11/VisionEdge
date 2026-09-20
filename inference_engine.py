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

print("Final input shape:", input_data.shape)
print("Data type:", input_data.dtype)

