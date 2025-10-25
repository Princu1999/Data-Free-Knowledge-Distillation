
import torch
import torch.nn.functional as F

def train_epoch(config, teacher, student, generator, device, optim_student, optim_gen):
    teacher.eval(); student.train(); generator.train()
    for _ in range(config.epoch_itrs):
        # Update student multiple steps
        for _ in range(15):
            z = torch.randn((config.batch_size, config.nz, 1, 1), device=device)
            fake = generator(z).detach()                    # 32x32
            fake_teacher = torch.nn.functional.interpolate(fake, size=224, mode='bilinear', align_corners=True)
            with torch.no_grad():
                t_logit = teacher(fake_teacher)
            s_logit = student(fake)
            loss_S = F.l1_loss(s_logit, t_logit)

            optim_student.zero_grad()
            loss_S.backward()
            optim_student.step()

        # Update generator once
        z = torch.randn((config.batch_size, config.nz, 1, 1), device=device)
        fake = generator(z)
        fake_teacher = torch.nn.functional.interpolate(fake, size=224, mode='bilinear', align_corners=True)
        t_logit = teacher(fake_teacher)
        s_logit = student(fake)
        loss_G = - F.l1_loss(s_logit, t_logit)

        optim_gen.zero_grad()
        loss_G.backward()
        optim_gen.step()
