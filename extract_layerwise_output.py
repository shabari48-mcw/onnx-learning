import onnxruntime as ort
import onnx
import numpy as np
from collections import OrderedDict

# file_name="scatternd.onnx"
# file_name="dynamic_update.onnx"
file_name="update_without_scatternd.onnx"

ort_session1 = ort.InferenceSession(file_name)

print([x.shape for x in ort_session1.get_inputs()])
input =  [np.random.randn(*x.shape).astype(np.float32) if i!=1 else np.random.randint(0,5,x.shape).astype(np.int64) for i,x in enumerate(ort_session1.get_inputs())]
inputs = dict(zip((x.name for x in ort_session1.get_inputs()),input))


org_outputs = [x.name for x in ort_session1.get_outputs()]

model = onnx.load(file_name)
for node in model.graph.node:
    for output in node.output:
        if output not in org_outputs:
            model.graph.output.extend([onnx.ValueInfoProto(name=output)])
 
ort_session = ort.InferenceSession(model.SerializeToString())

outputs = [x.name for x in ort_session.get_outputs()]

ort_outs = ort_session.run(None, input_feed=inputs)


from collections import OrderedDict
ort_outs = OrderedDict(zip(outputs, ort_outs))

print(ort_outs)