
#!/usr/bin/env python
import torch, os
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from akd_kd_gan.models.generator import GeneratorA

def main(out_dir: str = './generated_images', n: int = 25, nz: int = 100):
    os.makedirs(out_dir, exist_ok=True)
    G = GeneratorA(nz=nz, nc=3, img_size=32)
    # Optionally load weights if available at ./models/<run>/cifar100-generator.pt
    # G.load_state_dict(torch.load('.../cifar100-generator.pt', map_location='cpu'))

    with torch.no_grad():
        z = torch.randn(n, nz, 1, 1)
        fakes = G(z)
        fakes = torch.clamp((fakes + 1) / 2.0, 0.0, 1.0)
        grid = make_grid(fakes, nrow=int(n**0.5), padding=2, normalize=False)
        plt.figure(figsize=(8,8))
        plt.imshow(grid.permute(1,2,0).numpy()); plt.axis('off')
        plt.savefig(os.path.join(out_dir, 'grid.png'), bbox_inches='tight')
        # also save individual images
        for i in range(n):
            save_image(fakes[i], os.path.join(out_dir, f'image_{i:03d}.png'))

if __name__ == '__main__':
    main()
