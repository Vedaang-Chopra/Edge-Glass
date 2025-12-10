import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import List, Dict, Optional, Union
import torchvision
from torchvision import transforms

class ZeroShotClassifier:
    """Zero-shot classifier using vision-language alignment."""
    
    def __init__(self, model, device: str = "cuda"):
        self.model = model
        self.device = device
        self.class_embeddings = None
        self.class_names = None
        
    def build_classifier(self, class_names: List[str], templates: List[str]):
        """Build classifier weights from class names and templates.
        
        Args:
            class_names: List of class names (e.g., ["cat", "dog"])
            templates: List of prompt templates (e.g., ["a photo of a {}"])
        """
        self.class_names = class_names
        self.model.eval()
        
        all_embeddings = []
        
        with torch.no_grad():
            for name in tqdm(class_names, desc="Building classifier"):
                # Create prompts for this class
                prompts = [template.format(name) for template in templates]
                
                # Encode and average
                # Assuming model supports encode_text with list of strings
                if hasattr(self.model, "text_encoder"):
                    # Use the text encoder directly if possible
                    # We might need to handle tokenization depending on the specific encoder API
                    # The alignment model wrapper usually expects 'texts' argument
                    
                    # Batch the prompts if needed, but for building classifier usually small enough
                    # Check API: model.text_encoder(texts=...) -> returns output with .pooled
                     
                    # We use the model's forward path for convenience if available, or direct encoder
                    # Let's try to use the model's text encoder wrapper 
                    
                    if hasattr(self.model.text_encoder, "__call__"):
                        # Direct call
                        text_output = self.model.text_encoder(texts=prompts)
                        embeddings = text_output.pooled
                    else:
                        raise ValueError("Model does not have a callable text_encoder")
                        
                else:
                     raise ValueError("Model does not have text_encoder")

                # Normalize and average
                embeddings = F.normalize(embeddings, dim=-1)
                avg_embedding = embeddings.mean(dim=0)
                avg_embedding = F.normalize(avg_embedding, dim=-1)
                
                all_embeddings.append(avg_embedding)
                
        self.class_embeddings = torch.stack(all_embeddings).to(self.device) # (Num_Classes, Dim)
        
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate accuracy on a dataloader.
        
        Args:
            dataloader: DataLoader yielding (images, labels) or similar dict
            
        Returns:
            Dict with 'top1' and 'top5' accuracy
        """
        if self.class_embeddings is None:
            raise ValueError("Classifier not built. Call build_classifier first.")
            
        correct_1 = 0
        correct_5 = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Zero-shot evaluation"):
                # Handle different batch formats
                if isinstance(batch, dict):
                    images = batch.get("image") or batch.get("pixel_values")
                    labels = batch.get("label") or batch.get("labels")
                elif isinstance(batch, (list, tuple)):
                    images, labels = batch[0], batch[1]
                else:
                    continue
                    
                if images is None or labels is None:
                    continue
                    
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Get image embeddings
                if hasattr(self.model, "vision_encoder"):
                    image_out = self.model.vision_encoder(images)
                    image_features = image_out.pooled
                else:
                     # Fallback assuming model forward returns something usable or is the encoder
                     # Ideally we access vision_encoder directly
                     raise ValueError("Model missing vision_encoder")

                image_features = F.normalize(image_features, dim=-1)
                
                # Compute logits
                # (Batch, Dim) @ (Classes, Dim).T -> (Batch, Classes)
                logits = image_features @ self.class_embeddings.T
                
                # Calculate accuracy
                # Top-1
                pred_1 = logits.argmax(dim=-1)
                correct_1 += (pred_1 == labels).sum().item()
                
                # Top-5
                if logits.shape[1] >= 5:
                    _, pred_5 = logits.topk(5, dim=-1)
                    correct_5 += (pred_5 == labels.unsqueeze(1)).any(dim=-1).sum().item()
                else:
                    correct_5 += (pred_1 == labels).sum().item() # Fallback if < 5 classes
                    
                total += labels.size(0)
                
        if total == 0:
            return {"top1": 0.0, "top5": 0.0}
            
        return {
            "top1": 100 * correct_1 / total,
            "top5": 100 * correct_5 / total
        }

    @staticmethod
    def load_dataset(name: str, root_dir: str, split: str = "val", transform=None):
        """Helper to load standard torchvision datasets."""
        # Simple mapping for common datasets
        # Note: Users need to have these downloaded or use download=True if allowed/supported
        
        if name.lower() == "cifar100":
            return torchvision.datasets.CIFAR100(root=root_dir, train=(split=="train"), transform=transform, download=True)
        elif name.lower() == "cifar10":
            return torchvision.datasets.CIFAR10(root=root_dir, train=(split=="train"), transform=transform, download=True)   
        # Add others as needed, e.g., ImageNet often requires manual setup
        else:
            raise ValueError(f"Dataset {name} not directly supported in this helper yet.")
