import torch

print(torch.cuda.is_available())  # True가 나와야 GPU 사용 가능
print(torch.__version__)  # 버전 뒤에 +cu118 처럼 'cu'가 붙어 있어야 함
