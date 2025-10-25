
from dataclasses import dataclass, asdict
from datetime import datetime
import os, json, torch, numpy as np, random

@dataclass
class AKDConfig:
    # data
    dataset: str = "cifar100"
    data_root: str = "./data"
    image_size: int = 32
    test_split: float = 0.4

    # optimization
    epochs: int = 100
    epoch_itrs: int = 100
    batch_size: int = 256
    test_batch_size: int = 128
    lr_S: float = 0.01
    lr_G: float = 1e-3
    weight_decay: float = 5e-4
    momentum: float = 0.9
    scheduler: bool = False

    # kd / generator
    nz: int = 100
    temperature: float = 4.0
    alpha: float = 1.0
    beta: float = 1.5
    gamma: float = 2.0

    # ckpt / logging
    output_dir: str = "./output"
    model_save_dir: str = "./models"
    results_save_dir: str = "./results"
    log_dir: str = "./logs"
    checkpoint_freq: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    no_cuda: bool = False
    seed: int = 42
    teacher_model_path: str = "./teacher/best_resnet34_cifar100.pth"

    # populated at runtime
    run_id: str | None = None
    run_output_dir: str | None = None
    run_model_dir: str | None = None
    run_results_dir: str | None = None
    run_log_dir: str | None = None

    def prepare(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"akd_gan_run_{ts}"
        self.run_output_dir = os.path.join(self.output_dir, self.run_id)
        self.run_model_dir = os.path.join(self.model_save_dir, self.run_id)
        self.run_results_dir = os.path.join(self.results_save_dir, self.run_id)
        self.run_log_dir = os.path.join(self.log_dir, self.run_id)

        for d in [self.output_dir, self.model_save_dir, self.results_save_dir, self.log_dir,
                  self.run_output_dir, self.run_model_dir, self.run_results_dir, self.run_log_dir]:
            os.makedirs(d, exist_ok=True)

        # Reproducibility
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Persist config
        with open(os.path.join(self.run_log_dir, "config.json"), "w") as f:
            json.dump(asdict(self), f, indent=2)

        return self
