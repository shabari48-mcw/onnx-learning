import onnxruntime as ort
import onnx
import numpy as np
from collections import OrderedDict

file_name="./models/CNN.onnx"
ort_session_1 = ort.InferenceSession(file_name)
org_outputs = [x.name for x in ort_session_1.get_outputs()]

model = onnx.load(file_name)
for node in model.graph.node:
    for output in node.output:
        if output not in org_outputs:
            model.graph.output.extend([onnx.ValueInfoProto(name=output)])
 
ort_session = ort.InferenceSession(model.SerializeToString())
input_name  = ort_session.get_inputs()[0].name


output_names = [x.name for x in ort_session.get_outputs()]

input_shape = ort_session.get_inputs()[0].shape

input_data = np.random.randn(1, 3, 64, 64).astype(np.float32)

ort_outs = ort_session.run(output_names, {input_name: np.array(input_data)} )

from collections import OrderedDict
ort_outs = OrderedDict(zip(output_names, ort_outs))

arr1=ort_outs['/conv_block_2/conv_block_2.2/Conv_output_0']

print(arr1.shape)