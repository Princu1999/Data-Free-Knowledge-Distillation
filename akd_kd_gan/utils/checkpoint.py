
import torch

def save_checkpoint(model, optimizer, epoch, loss, filename):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filename)

def load_checkpoint(model, optimizer, filename):
    ckpt = torch.load(filename, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt.get('epoch', 0), ckpt.get('loss', None)
