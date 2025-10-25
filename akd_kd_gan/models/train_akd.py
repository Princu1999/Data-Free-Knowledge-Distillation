
#!/usr/bin/env python
import torch, os
from akd_kd_gan.config import AKDConfig
from akd_kd_gan.models.generator import GeneratorA
from akd_kd_gan.models.resnet_small import ResNet18_8x, ResNet18_8x_Small
from akd_kd_gan.data import load_data_cifar100
from akd_kd_gan.engine.train import train_epoch
from akd_kd_gan.engine.eval import evaluate
import torch.optim as optim
from torchvision import models

def build_teacher(model_path: str, num_classes: int = 100, device: str = "cpu"):
    model = models.resnet34(weights=None, num_classes=num_classes)
    try:
        ckpt = torch.load(model_path, map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        else:
            model.load_state_dict(ckpt)
        print("Loaded teacher weights from", model_path)
    except Exception as e:
        print("WARNING: could not load teacher weights:", e)
    return model

def main():
    cfg = AKDConfig().prepare()
    device = torch.device("cuda" if (not cfg.no_cuda and torch.cuda.is_available()) else "cpu")

    # data
    test_loader = load_data_cifar100(cfg.data_root, cfg.batch_size, cfg.test_split)

    # models
    teacher = build_teacher(cfg.teacher_model_path, num_classes=100, device=cfg.device).to(device).eval()
    student_50 = ResNet18_8x(num_classes=100).to(device)
    student_20 = ResNet18_8x_Small(num_classes=100).to(device)
    generator = GeneratorA(nz=cfg.nz, nc=3, img_size=32).to(device)

    # optim
    opt_S50 = optim.SGD(student_50.parameters(), lr=cfg.lr_S, weight_decay=cfg.weight_decay, momentum=cfg.momentum)
    opt_S20 = optim.SGD(student_20.parameters(), lr=cfg.lr_S, weight_decay=cfg.weight_decay, momentum=cfg.momentum)
    opt_G   = optim.Adam(generator.parameters(), lr=cfg.lr_G)

    best50 = best20 = 0.0
    acc_hist_50, acc_hist_20 = [], []

    for epoch in range(1, cfg.epochs + 1):
        # student 50%
        train_epoch(cfg, teacher, student_50, generator, device, opt_S50, opt_G)
        # student 20%
        train_epoch(cfg, teacher, student_20, generator, device, opt_S20, opt_G)

        # eval
        loss50, acc50 = evaluate(student_50, device, test_loader)
        loss20, acc20 = evaluate(student_20, device, test_loader)

        acc_hist_50.append(acc50); acc_hist_20.append(acc20)
        print(f"[Epoch {epoch}] acc50={acc50:.4f} acc20={acc20:.4f}")

        # save best
        if acc50 > best50:
            best50 = acc50
            torch.save(student_50.state_dict(), os.path.join(cfg.run_model_dir, f"{cfg.dataset}-resnet18_8x_50p.pt"))
        if acc20 > best20:
            best20 = acc20
            torch.save(student_20.state_dict(), os.path.join(cfg.run_model_dir, f"{cfg.dataset}-resnet18_8x_20p.pt"))
        torch.save(generator.state_dict(), os.path.join(cfg.run_model_dir, f"{cfg.dataset}-generator.pt"))

    # persist history
    import csv
    with open(os.path.join(cfg.run_log_dir, 'akd_results.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['epoch','acc_50','acc_20'])
        for i, (a,b) in enumerate(zip(acc_hist_50, acc_hist_20), start=1):
            w.writerow([i, a, b])

    print(f"Best Acc (50%): {best50:.4f}  |  Best Acc (20%): {best20:.4f}")

if __name__ == '__main__':
    main()
