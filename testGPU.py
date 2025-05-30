import torch

# Check if PyTorch is installed and CUDA is available
print(torch.version.cuda)			# CUDA version
print(torch.cuda.is_available())	# True if GPU available
print(torch.cuda.get_device_name(0)) # GPU name
