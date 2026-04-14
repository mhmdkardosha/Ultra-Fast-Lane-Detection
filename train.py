import torch, os, datetime
import numpy as np
import re

from model.model import parsingNet
from data.dataloader import get_train_loader, get_val_loader

from utils.dist_utils import dist_print, dist_tqdm, is_main_process, DistSummaryWriter
from utils.factory import get_metric_dict, get_loss_dict, get_optimizer, get_scheduler
from utils.metrics import (
    MultiLabelAcc,
    AccTopk,
    Metric_mIoU,
    update_metrics,
    reset_metrics,
)

from utils.common import merge_config, cp_projects
from utils.common import get_work_dir, get_logger

import time


def inference(net, data_label, use_aux):
    if use_aux:
        img, cls_label, seg_label = data_label
        img, cls_label, seg_label = (
            img.cuda(),
            cls_label.long().cuda(),
            seg_label.long().cuda(),
        )
        cls_out, seg_out = net(img)
        return {
            "cls_out": cls_out,
            "cls_label": cls_label,
            "seg_out": seg_out,
            "seg_label": seg_label,
        }
    else:
        img, cls_label = data_label
        img, cls_label = img.cuda(), cls_label.long().cuda()
        cls_out = net(img)
        return {"cls_out": cls_out, "cls_label": cls_label}


def resolve_val_data(results, use_aux):
    results["cls_out"] = torch.argmax(results["cls_out"], dim=1)
    if use_aux:
        results["seg_out"] = torch.argmax(results["seg_out"], dim=1)
    return results


def _format_metric_summary(metric_dict):
    if len(metric_dict) == 0:
        return ""
    return ", ".join(["%s=%.4f" % (k, v) for k, v in metric_dict.items()])


def _save_checkpoint(path, net, optimizer, epoch, extra=None):
    model_state_dict = net.state_dict()
    state = {
        "model": model_state_dict,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    if extra is not None:
        state.update(extra)
    torch.save(state, path)


def _infer_resume_epoch(resume_path, resume_dict):
    if "epoch" in resume_dict:
        return int(resume_dict["epoch"]) + 1

    file_name = os.path.basename(resume_path)
    matched = re.search(r"ep(\d+)", file_name)
    if matched is not None:
        return int(matched.group(1)) + 1

    return 0


def _select_monitor_metric(val_stats):
    metrics = val_stats.get("metrics", {})
    for metric_name in ["laneiou", "iou", "top1"]:
        if metric_name in metrics:
            return metric_name, float(metrics[metric_name]), "max"
    return "loss", float(val_stats["loss"]), "min"


def calc_loss(
    loss_dict, results, logger, global_step, log_prefix="loss", log_interval=20
):
    loss = 0

    for i in range(len(loss_dict["name"])):
        data_src = loss_dict["data_src"][i]

        datas = [results[src] for src in data_src]

        loss_cur = loss_dict["op"][i](*datas)

        if global_step % log_interval == 0:
            logger.add_scalar(
                log_prefix + "/" + loss_dict["name"][i],
                loss_cur.detach().item(),
                global_step,
            )

        loss += loss_cur * loss_dict["weight"][i]
    return loss


def train(
    net,
    data_loader,
    loss_dict,
    optimizer,
    scheduler,
    logger,
    epoch,
    metric_dict,
    use_aux,
    step_offset,
    log_interval,
):
    net.train()
    progress_bar = dist_tqdm(data_loader)
    t_data_0 = time.time()
    loss_sum = 0.0
    metric_sum = {name: 0.0 for name in metric_dict["name"]}
    num_batches = 0
    for b_idx, data_label in enumerate(progress_bar):
        t_data_1 = time.time()
        reset_metrics(metric_dict)
        # Keep scheduler steps based only on training iterations.
        train_step = epoch * len(data_loader) + b_idx
        # Keep logging steps monotonic across train+val in every epoch.
        log_step = step_offset + b_idx

        t_net_0 = time.time()
        results = inference(net, data_label, use_aux)

        loss = calc_loss(
            loss_dict,
            results,
            logger,
            log_step,
            log_prefix="loss",
            log_interval=log_interval,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(train_step)
        t_net_1 = time.time()

        results = resolve_val_data(results, use_aux)

        update_metrics(metric_dict, results)
        current_metrics = {
            me_name: float(me_op.get())
            for me_name, me_op in zip(metric_dict["name"], metric_dict["op"])
        }
        for me_name, me_value in current_metrics.items():
            metric_sum[me_name] += me_value
        loss_sum += loss.detach().item()
        num_batches += 1

        if log_step % log_interval == 0:
            for me_name, me_op in zip(metric_dict["name"], metric_dict["op"]):
                logger.add_scalar(
                    "metric/" + me_name, me_op.get(), global_step=log_step
                )
        logger.add_scalar(
            "meta/lr", optimizer.param_groups[0]["lr"], global_step=log_step
        )

        if hasattr(progress_bar, "set_postfix"):
            kwargs = {
                me_name: "%.3f" % current_metrics[me_name]
                for me_name in metric_dict["name"]
            }
            progress_bar.set_postfix(
                loss="%.3f" % loss.detach().item(),
                d_time="%.3f" % float(t_data_1 - t_data_0),
                n_time="%.3f" % float(t_net_1 - t_net_0),
                **kwargs,
            )
        t_data_0 = time.time()

    denom = max(1, num_batches)
    return {
        "loss": loss_sum / denom,
        "metrics": {name: metric_sum[name] / denom for name in metric_sum},
    }


def validate(
    net,
    data_loader,
    loss_dict,
    logger,
    epoch,
    metric_dict,
    use_aux,
    step_offset,
    log_interval,
):
    net.eval()
    progress_bar = dist_tqdm(data_loader)
    loss_sum = 0.0
    metric_sum = {name: 0.0 for name in metric_dict["name"]}
    num_batches = 0

    with torch.no_grad():
        for b_idx, data_label in enumerate(progress_bar):
            reset_metrics(metric_dict)
            global_step = step_offset + b_idx

            results = inference(net, data_label, use_aux)
            loss = calc_loss(
                loss_dict,
                results,
                logger,
                global_step,
                log_prefix="val_loss",
                log_interval=log_interval,
            )

            results = resolve_val_data(results, use_aux)
            update_metrics(metric_dict, results)

            current_metrics = {
                me_name: float(me_op.get())
                for me_name, me_op in zip(metric_dict["name"], metric_dict["op"])
            }
            for me_name, me_value in current_metrics.items():
                metric_sum[me_name] += me_value

            loss_sum += loss.detach().item()
            num_batches += 1

            if global_step % log_interval == 0:
                for me_name, me_value in current_metrics.items():
                    logger.add_scalar(
                        "metric/val_" + me_name, me_value, global_step=global_step
                    )

            if hasattr(progress_bar, "set_postfix"):
                kwargs = {
                    me_name: "%.3f" % current_metrics[me_name]
                    for me_name in metric_dict["name"]
                }
                progress_bar.set_postfix(
                    val_loss="%.3f" % loss.detach().item(), **kwargs
                )

    denom = max(1, num_batches)
    return {
        "loss": loss_sum / denom,
        "metrics": {name: metric_sum[name] / denom for name in metric_sum},
    }


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    work_dir = get_work_dir(cfg)

    distributed = False
    if "WORLD_SIZE" in os.environ:
        distributed = int(os.environ["WORLD_SIZE"]) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    wandb_run_id = None
    resume_dict = None

    if cfg.resume is not None:
        dist_print("==> Resume model from " + cfg.resume)
        resume_dict = torch.load(cfg.resume, map_location="cpu")
        wandb_run_id = resume_dict.get("wandb_run_id")

    if wandb_run_id is None:
        wandb_run_id = os.environ.get("WANDB_RUN_ID")

    dist_print(
        datetime.datetime.now().strftime("[%Y/%m/%d %H:%M:%S]") + " start training..."
    )
    if is_main_process():
        try:
            import wandb

            # Convert cfg to dict carefully for wandb config
            cfg_dict = {k: v for k, v in cfg.__dict__.items() if not k.startswith("__")}
            if wandb_run_id is not None:
                wandb.init(
                    project="UFLD-TuLane",
                    config=cfg_dict,
                    id=wandb_run_id,
                    resume="must",
                )
                dist_print("Resumed wandb run id:", wandb_run_id)
            else:
                wandb.init(project="UFLD-TuLane", config=cfg_dict)
                if wandb.run is not None:
                    wandb_run_id = wandb.run.id
        except ImportError:
            dist_print("wandb not installed, skipping wandb logging")
    dist_print(cfg)
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

    train_loader, cls_num_per_lane = get_train_loader(
        cfg.batch_size,
        cfg.data_root,
        cfg.griding_num,
        cfg.dataset,
        cfg.use_aux,
        distributed,
        cfg.num_lanes,
    )
    val_loader = get_val_loader(
        cfg.batch_size,
        cfg.data_root,
        cfg.griding_num,
        cfg.dataset,
        cfg.use_aux,
        distributed,
        cfg.num_lanes,
    )

    net = parsingNet(
        pretrained=True,
        backbone=cfg.backbone,
        cls_dim=(cfg.griding_num + 1, cls_num_per_lane, cfg.num_lanes),
        use_aux=cfg.use_aux,
    ).cuda()

    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[args.local_rank]
        )
    optimizer = get_optimizer(net, cfg)

    if cfg.finetune is not None:
        dist_print("finetune from ", cfg.finetune)
        state_all = torch.load(cfg.finetune)["model"]
        state_clip = {}  # only use backbone parameters
        for k, v in state_all.items():
            if "model" in k:
                state_clip[k] = v
        net.load_state_dict(state_clip, strict=False)

    if resume_dict is not None:
        net.load_state_dict(resume_dict["model"])
        if "optimizer" in resume_dict.keys():
            optimizer.load_state_dict(resume_dict["optimizer"])
        resume_epoch = _infer_resume_epoch(cfg.resume, resume_dict)
        dist_print("Resume epoch start:", resume_epoch)
    else:
        resume_epoch = 0

    scheduler = get_scheduler(optimizer, cfg, len(train_loader))
    dist_print(len(train_loader))
    train_metric_dict = get_metric_dict(cfg)
    val_metric_dict = get_metric_dict(cfg)
    loss_dict = get_loss_dict(cfg)
    logger = get_logger(work_dir, cfg)
    cp_projects(args.auto_backup, work_dir)

    best_metric_name = None
    best_metric_value = None
    best_epoch = -1

    if resume_dict is not None:
        best_metric_name = resume_dict.get(
            "best_metric_name", resume_dict.get("monitor_name")
        )
        best_metric_value = resume_dict.get(
            "best_metric_value", resume_dict.get("monitor_value")
        )
        best_epoch = int(resume_dict.get("best_epoch", resume_epoch - 1))

    train_steps_per_epoch = len(train_loader)
    val_steps_per_epoch = len(val_loader)
    log_steps_per_epoch = train_steps_per_epoch + val_steps_per_epoch
    log_interval = max(1, int(getattr(cfg, "batch_log_interval", 20)))

    for epoch in range(resume_epoch, cfg.epoch):
        train_step_offset = epoch * log_steps_per_epoch
        train_stats = train(
            net,
            train_loader,
            loss_dict,
            optimizer,
            scheduler,
            logger,
            epoch,
            train_metric_dict,
            cfg.use_aux,
            step_offset=train_step_offset,
            log_interval=log_interval,
        )
        val_stats = validate(
            net,
            val_loader,
            loss_dict,
            logger,
            epoch,
            val_metric_dict,
            cfg.use_aux,
            step_offset=train_step_offset + train_steps_per_epoch,
            log_interval=log_interval,
        )

        epoch_log_step = epoch + 1

        logger.add_scalar(
            "epoch/train_loss", train_stats["loss"], global_step=epoch_log_step
        )
        logger.add_scalar(
            "epoch/val_loss", val_stats["loss"], global_step=epoch_log_step
        )
        for me_name, me_value in train_stats["metrics"].items():
            logger.add_scalar(
                "epoch/train_" + me_name, me_value, global_step=epoch_log_step
            )
        for me_name, me_value in val_stats["metrics"].items():
            logger.add_scalar(
                "epoch/val_" + me_name, me_value, global_step=epoch_log_step
            )

        monitor_name, monitor_value, monitor_mode = _select_monitor_metric(val_stats)
        is_best = False
        if best_metric_value is None:
            is_best = True
        elif monitor_mode == "max":
            is_best = monitor_value > best_metric_value
        else:
            is_best = monitor_value < best_metric_value

        if is_best:
            best_metric_name = monitor_name
            best_metric_value = monitor_value
            best_epoch = epoch

        if is_main_process():
            ckpt_extra = {
                "monitor_name": monitor_name,
                "monitor_value": monitor_value,
                "best_metric_name": best_metric_name,
                "best_metric_value": best_metric_value,
                "best_epoch": best_epoch,
                "wandb_run_id": wandb_run_id,
            }

            _save_checkpoint(
                os.path.join(work_dir, "latest.pth"),
                net,
                optimizer,
                epoch,
                extra=ckpt_extra,
            )

            if is_best:
                _save_checkpoint(
                    os.path.join(work_dir, "best.pth"),
                    net,
                    optimizer,
                    epoch,
                    extra=ckpt_extra,
                )

        dist_print(
            "[Epoch %d/%d] train_loss=%.4f, val_loss=%.4f, train{%s}, val{%s}, monitor=%s:%.4f, best=%s:%.4f@ep%d"
            % (
                epoch + 1,
                cfg.epoch,
                train_stats["loss"],
                val_stats["loss"],
                _format_metric_summary(train_stats["metrics"]),
                _format_metric_summary(val_stats["metrics"]),
                monitor_name,
                monitor_value,
                best_metric_name,
                best_metric_value,
                best_epoch + 1,
            )
        )

    logger.close()
