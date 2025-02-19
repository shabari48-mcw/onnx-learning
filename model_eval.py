import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import onnxruntime as ort
from torch.utils.data import Subset
from tqdm import tqdm


class ModelEvaluator:
    def __init__(self, model_path:str, device :str ='cuda'):
        """ Initialize the ModelEvaluator class."""
        self.device = device
        self.ort_session = ort.InferenceSession(model_path)
        print(ort.get_device())
        

    def get_transforms(self):
        """ Return the transformed image tensor."""
        return transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

    def evaluate(self, data_path: str, num_classes :int =50, images_per_class :int =5, batch_size:int =64) -> dict :
        
        # Prepare dataset
        dataset = ImageFolder(
            root=data_path,
            transform=self.get_transforms(),
        )
        
        all_classes=dataset.classes
        
        # Filter dataset to include only a subset of classes
        selected_classes = random.sample(all_classes, num_classes)
        
        # Create a mapping of selected class names to their indices
        selected_class_indices = [dataset.class_to_idx[class_name] for class_name in selected_classes]
        
        # Get indices of samples belonging to selected classes
        selected_indices = []
        class_counts = {idx: 0 for idx in selected_class_indices}
        
        for idx, (_, label) in enumerate(dataset.samples):
            if label in selected_class_indices and class_counts[label] < images_per_class:
                selected_indices.append(idx)
                class_counts[label] += 1
                
        # Create a subset dataset
        subset_dataset = Subset(dataset, selected_indices)
        
        
        dataloader = DataLoader(
            subset_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )

        top1_correct = 0
        top5_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Evaluating"):
             
                # ONNX  Model inference
                ort_inputs = {
                    self.ort_session.get_inputs()[0].name: 
                    images.numpy()
                }
                outputs = self.ort_session.run(None, ort_inputs)[0]
                outputs = torch.FloatTensor(outputs)

                # Calculate Top-1 and Top-5 accuracy
                _, pred = outputs.topk(5, 1, True, True)
                pred = pred.t()
                labels = labels.view(1, -1).expand_as(pred)
                
                correct = pred.eq(labels)
             
                top1_correct += correct[0].sum().item()
                top5_correct += correct[:5].any(0).sum().item()
                total_samples += labels.size(1)

        # Calculate final metrics
        top1_accuracy = (top1_correct / total_samples) * 100
        top5_accuracy = (top5_correct / total_samples) * 100

        return {
            'top1_accuracy': top1_accuracy,
            'top5_accuracy': top5_accuracy,
            'total_samples': total_samples
        }
        

def main():
   
    data_path = '/media/bmw/datasets/imagenet-1k/val'  

    # Evaluate ONNX model
    onnx_evaluator = ModelEvaluator(
        model_path='resnet50.onnx'
    )
    
    onnx_results = onnx_evaluator.evaluate(
        data_path,
        num_classes=1000,
        images_per_class=5
    )
    
    # Print results of Evaluation
    print("\nONNX Model Results:")
    print(f"Top-1 Accuracy: {onnx_results['top1_accuracy']:.2f}%")
    print(f"Top-5 Accuracy: {onnx_results['top5_accuracy']:.2f}%")
    print(f"Total Samples Evaluated: {onnx_results['total_samples']}")


if __name__ == '__main__':
    main()