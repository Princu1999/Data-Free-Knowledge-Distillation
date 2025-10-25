
import torch
import torch.nn.functional as F

def kd_kl_divergence(student_logits, teacher_logits, temperature: float = 4.0):
    eps = 1e-8
    s_log_softmax = F.log_softmax(student_logits / temperature, dim=1)
    t_softmax = F.softmax(teacher_logits / temperature, dim=1) + eps
    kl = F.kl_div(s_log_softmax, t_softmax, reduction='batchmean', log_target=False)
    return kl * (temperature ** 2)

def diversity_loss(images: torch.Tensor):
    bs = images.size(0)
    flat = images.view(bs, -1)
    norm = F.normalize(flat, p=2, dim=1)
    sim = torch.mm(norm, norm.t())
    sim = sim * (1 - torch.eye(bs, device=images.device))
    return sim.sum() / (bs * (bs - 1))
