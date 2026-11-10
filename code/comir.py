"""CoMIR baseline (Pielawski et al., NeurIPS 2020): one tiramisu per modality maps images to a shared representation,
trained contrastively on aligned pairs (InfoNCE, MSE critic, tau 0.1, C4 equivariance), settings from the authors'
train-zurich notebook. Registration: exhaustive integer search (+-24 px) of the mean squared difference between the
two representations over the overlap, the same search range as the other baselines."""
import sys

import torch

LATENT, RADIUS = 3, 24
TIRAMISU = dict(init_conv_filters=16, down_blocks=(2, 2, 2), up_blocks=(2, 2, 2), bottleneck_layers=3, upsampling_type="upsample",
                transition_pooling="max", dropout_rate=0.0, early_transition=False, activation_func=None, compression=1.0, efficient=False)


def networks(repo, seed=0):
    sys.path.insert(0, repo)
    from models.tiramisu import DenseUNet

    class ModNet(DenseUNet):
        def __init__(self, **args):
            super().__init__(**args, include_top=False)
            self.final_conv = torch.nn.Conv2d(self.get_channels_count()[-1], LATENT, 1, bias=False)

        def forward(self, x):
            return self.final_conv(super().forward(x))

    torch.manual_seed(seed)  # the notebook fixes seed 0; other seeds give independent initialisations
    return torch.nn.ModuleList([ModNet(in_channels=1, nb_classes=LATENT, **TIRAMISU) for _ in range(2)])


def load(path, repo, device):
    nets = networks(repo)
    nets.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return nets.to(device).eval()


def represent(nets, x):
    """x [B, 2, 3, H, W] -> [B, 2, LATENT, H, W]; each stream's first channel goes through its own network."""
    return torch.stack([nets[m](x[:, m, :1]) for m in range(2)], 1)


def search(ref, mov, radius=RADIUS):
    """Integer d per item minimising the mean squared difference of ref[y] and mov[y - d] over the overlap. ref, mov [B, C, H, W]."""
    b, _, h, w = ref.shape
    best = torch.full((b,), float("inf"), device=ref.device)
    arg = torch.zeros(b, 2, dtype=torch.long, device=ref.device)
    for dy in range(-radius, radius + 1):
        r = ref[:, :, max(dy, 0):h + min(dy, 0)]
        m = mov[:, :, max(-dy, 0):h + min(-dy, 0)]
        for dx in range(-radius, radius + 1):
            score = (r[..., max(dx, 0):w + min(dx, 0)] - m[..., max(-dx, 0):w + min(-dx, 0)]).pow(2).mean((1, 2, 3))
            better = score < best
            best = torch.where(better, score, best)
            arg[better] = torch.tensor([dy, dx], device=ref.device)
    return [tuple(int(v) for v in d) for d in arg.cpu()]


@torch.inference_mode()
def shifts(nets, scene, target, batch, loaders, device):
    from cca_eval import collate

    out = []
    for x in torch.utils.data.DataLoader(scene, batch, num_workers=loaders, collate_fn=collate):
        z = represent(nets, x.to(device)).float()
        out += search(z[:, 1 - target], z[:, target])
    return out
