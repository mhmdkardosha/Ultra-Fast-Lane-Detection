import torch, os
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
from evaluation.eval_wrapper import eval_lane

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = torch.cuda.is_available()

    args, cfg = merge_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dist_print("start testing (python-only mode)...")
    dist_print(f"device: {device}")
    assert cfg.backbone in [
        "18",
        "34",
        "50",
        "101",
        "152",
        "50next",
        "101next",
        "50wide",
        "101wide",
    ]

    if cfg.dataset == "CULane":
        cls_num_per_lane = 18
    elif cfg.dataset == "Tusimple":
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    net = parsingNet(
        pretrained=False,
        backbone=cfg.backbone,
        cls_dim=(cfg.griding_num + 1, cls_num_per_lane, cfg.num_lanes),
        use_aux=False,
    ).to(device)

    state_dict = torch.load(cfg.test_model, map_location="cpu")["model"]
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if "module." in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)

    if not os.path.exists(cfg.test_work_dir):
        os.mkdir(cfg.test_work_dir)

    eval_lane(
        net,
        cfg.dataset,
        cfg.data_root,
        cfg.test_work_dir,
        cfg.griding_num,
        False,
        distributed=False,
    )
