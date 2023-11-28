import torch

print("\n=============================")
print("torch.__version__: " + str(torch.__version__))
print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
if torch.cuda.is_available():
    print("torch.cuda.get_device_name(0): " + str(torch.cuda.get_device_name(0)))
print("=============================\n")
