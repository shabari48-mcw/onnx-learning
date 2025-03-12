import onnxruntime as ort
import onnx
import numpy as np
from collections import OrderedDict

file_name=r"D:\Learn DL\ONNX\scatternd.onnx"

data_np = np.array([[0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0]], dtype=np.float32)

indices_np = np.array([[1, 0, 2],
                            [0, 2, 1]], dtype=np.int64)

updates_np = np.array([[1.0, 1.1, 1.2],
                       [2.0, 2.1, 2.2]], dtype=np.float32)


ort_session_1 = ort.InferenceSession(file_name)
org_outputs = [x.name for x in ort_session_1.get_outputs()]

model = onnx.load(file_name)
for node in model.graph.node:
    for output in node.output:
        if output not in org_outputs:
            model.graph.output.extend([onnx.ValueInfoProto(name=output)])
 
ort_session = ort.InferenceSession(model.SerializeToString())

outputs = [x.name for x in ort_session.get_outputs()]
inputs = [x.name for x in ort_session.get_inputs()]
input_shape = ort_session.get_inputs()[0].shape



ort_outs = ort_session.run(None, input_feed={
    "data": data_np,
    "indices": indices_np,
    "updates": updates_np,
})


from collections import OrderedDict
ort_outs = OrderedDict(zip(outputs, ort_outs))

print(ort_outs)