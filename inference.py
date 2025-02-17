import onnx
import onnxruntime as ort
import onnx_graphsurgeon as gs
import numpy as np
from scipy.special import softmax
from pathlib import Path
import argparse
import time
from typing import List, Tuple , Dict


def create_session(model_path: str) -> Tuple[ort.InferenceSession, List[Tuple[str, Tuple[int]]], List[Tuple[str, Tuple[int]]]]:

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    onnx_session = ort.InferenceSession(model_path)
    graph = gs.import_onnx(onnx.load(model_path))
    inputs = [(input.name, input.shape) for input in graph.inputs]
    outputs = [(output.name, output.shape) for output in graph.outputs]
    return onnx_session, inputs, outputs


def generate_random_inputs(
    input_shapes: List[Tuple[str, Tuple[int]]], num_inputs: int
) -> List[Dict[str, np.ndarray]]:
    """Generate multiple random inputs based on the input shapes"""
    input_list = []
    for _ in range(num_inputs):
        inputs = {}
        for name, shape in input_shapes:
            random_input = np.random.rand(*shape).astype(np.float32)
            inputs[name] = random_input
        input_list.append(inputs)
    return input_list


def run_batch_inference(
    onnx_session,
    input_data : Dict[str, np.ndarray],
    output_data: List[str],
) -> List[List[np.ndarray]]:
    
    all_outputs = []
    total_time = 0
    
    for idx, input_data in enumerate(input_list):
        start = time.time()
        outputs = onnx_session.run(output_names, input_data)
        end = time.time()
        inference_time = end - start
        total_time += inference_time
        
        print(f"\nInput batch {idx+1}:")
        print(f"Inference time: {inference_time:.4f} seconds")
        all_outputs.append(outputs)
    
    print(f"\nTotal inference time: {total_time:.4f} seconds")
    print(f"Average inference time: {total_time/len(input_list):.4f} seconds")
    return all_outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on ONNX model")
    parser.add_argument("-m", "--model_path", type=str, help="Path to ONNX model")
    parser.add_argument("-n", "--num_inputs", type=int, help="NUmber of input data",default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    MODEL_PATH = Path(args.model_path)
    NUM_INPUTS = max(1, args.num_inputs)  

    print(f"Model Path: {MODEL_PATH}")
    print(f"Number of input batches: {NUM_INPUTS}")
    
    print("Running inference..")

    onnx_session, inputs, outputs = create_session(MODEL_PATH)
    
    for input in inputs:
        print(f"Input: {input[0]} Shape: {input[1]}")
        
    print()
    for output in outputs:
        print(f"Output: {output[0]} Shape: {output[1]}")

    input_list = generate_random_inputs(inputs,NUM_INPUTS)
    
    print(f"\nGenerated {NUM_INPUTS} random input batches")
  
    output_names = [output.name for output in onnx_session.get_outputs()]
    

    all_outputs = run_batch_inference(onnx_session, input_list, output_names)

    print("\nInference completed")
