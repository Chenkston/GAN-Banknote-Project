import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

class NoteShieldDataset(Dataset):
    """
    Data loader for the NoteShieldBD (JaalTaka) dataset.
    Handles loading 6 individual segment images (.jpg) per banknote and stacking them into a tensor.
    """
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Path to the directory containing the dataset.
                            Assumes structure:
                            data_dir/
                                real_notes/
                                    note_001/
                                        note_001_1.jpg
                                        note_001_2.jpg
                                        ...
                                        note_001_6.jpg
                                fake_notes/
                                    note_001/
                                        ...
            transform (callable, optional): Optional transform to be applied on the stacked tensor.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []  # List of paths to note directories (e.g., .../real_notes/note_001)
        self.labels = []
        self.cache = {}    # Cache tensors to save from having to load 6 images every time
        
        # Expected class subdirectories mapping to labels
        # 0 for real (genuine), 1 for fake (counterfeit)
        self.classes = {'real_notes': 0, 'fake_notes': 1}
        
        for class_name, label in self.classes.items():
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                # Iterate over note directories (e.g., note_001, note_002)
                for note_dir_name in sorted(os.listdir(class_dir)):
                    note_dir_path = os.path.join(class_dir, note_dir_name)
                    if os.path.isdir(note_dir_path):
                        self.samples.append(note_dir_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Return cached result if available
        if idx in self.cache:
            return self.cache[idx]

        note_dir = self.samples[idx]
        label = self.labels[idx]

        # Get all image paths in the note directory and sort them alphabetically 
        # to ensure order (_1.jpg, _2.jpg, ..., _6.jpg)
        image_files = sorted([f for f in os.listdir(note_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        if len(image_files) != 6:
            raise ValueError(f"Expected exactly 6 images in {note_dir}, found {len(image_files)}")

        segments = []
        for img_file in image_files:
            img_path = os.path.join(note_dir, img_file)
            # Load image and convert to RGB (ensures 3 channels even if grayscale is present)
            img = Image.open(img_path).convert('RGB')
            # Resize images to 64 x 64, the input size for our custom models
            img = img.resize((64, 64))
            # Convert PIL Image to PyTorch tensor (C, H, W) with values in range [0.0, 1.0]
            img_tensor = TF.to_tensor(img)
            segments.append(img_tensor)
            
        # Stack the 6 image tensors along a new first dimension: shape becomes (6, C, H, W)
        segments_tensor = torch.stack(segments)

        if self.transform:
            # Note: Transform should be capable of handling (6, C, H, W) inputs
            segments_tensor = self.transform(segments_tensor)

        result = (segments_tensor, label)
    
        # Store in cache before returning
        self.cache[idx] = result

        return result
