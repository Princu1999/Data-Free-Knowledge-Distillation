
import torch
import torch.nn.functional as F

@torch.no_grad()
def evaluate(student, device, test_loader):
    student.eval()
    test_loss = 0; correct = 0; total = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = student(data)
        test_loss += F.cross_entropy(output, target, reduction='sum').item()
        pred = output.argmax(dim=1, keepdim=False)
        correct += (pred == target).sum().item()
        total += target.size(0)
    return test_loss / total, correct / total
