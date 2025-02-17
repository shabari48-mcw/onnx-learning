# %%
import onnx
import onnxruntime as ort
import numpy as np
import onnx_graphsurgeon as gs
import os

# %%
CNN = "./models/CNN.onnx"

# %%
model = onnx.load(CNN)
graph = gs.import_onnx(model)


# %%
graph.inputs, graph.outputs


# %%
inter_nodes = [nodes.name for nodes in graph.nodes]
inter_nodes


# %%
def add_intermediate_outputs(model_path):

    model = onnx.load(model_path)
    graph = gs.import_onnx(model)

    # Add intermediate outputs
    for node in graph.nodes:
        if node.name in inter_nodes:

            output_name = f"{node.name}_output"

            original_output = node.outputs[0]
            dtype = original_output.dtype
            shape = original_output.shape

            new_tensor = gs.Variable(
                name=output_name,
                dtype=dtype if dtype is not None else np.float32,
                shape=shape,
            )

            graph.outputs.append(new_tensor)

            identity_node = gs.Node(
                op="Identity",
                name=f"{node.name}_identity",
                inputs=[node.outputs[0]],
                outputs=[new_tensor],
            )

            graph.nodes.append(identity_node)

    modified_model_path = model_path.replace(".onnx", "_with_outputs.onnx")
    onnx.save(gs.export_onnx(graph), modified_model_path)
    print(f"Modified model saved with  intermediate outputs")
    return modified_model_path


def save_layer_outputs(session, input_data, output_dir="layer_outputs"):
    
    os.makedirs(output_dir, exist_ok=True)

    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]

    outputs = session.run(output_names, {input_name: input_data})

    for layer_name, output in zip(output_names, outputs):
        layer_name = layer_name.replace("/", "_")
        output_path = os.path.join(output_dir, f"{layer_name}.raw")
        output.tofile(output_path)
        print(f"Saved {layer_name} with shape {output.shape} to {output_path}")


if __name__ == "__main__":

    CNN = "./models/CNN.onnx"

    modified_model_path = add_intermediate_outputs(CNN)

    session = ort.InferenceSession(modified_model_path)

    input_data = np.random.randn(1, 3, 64, 64).astype(np.float32)

    save_layer_outputs(session, input_data)
