from data.dataloader import get_test_loader
from evaluation.tusimple.lane import LaneEval
from utils.dist_utils import (
    is_main_process,
    dist_print,
    get_rank,
    get_world_size,
    dist_tqdm,
    synchronize,
)
import os, json, torch, scipy
import numpy as np


def generate_lines(
    out,
    shape,
    names,
    output_path,
    griding_num,
    localization_type="abs",
    flip_updown=False,
):

    col_sample = np.linspace(0, shape[1] - 1, griding_num)
    col_sample_w = col_sample[1] - col_sample[0]

    for j in range(out.shape[0]):
        out_j = out[j].data.cpu().numpy()
        if flip_updown:
            out_j = out_j[:, ::-1, :]
        if localization_type == "abs":
            out_j = np.argmax(out_j, axis=0)
            out_j[out_j == griding_num] = -1
            out_j = out_j + 1
        elif localization_type == "rel":
            prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
            idx = np.arange(griding_num) + 1
            idx = idx.reshape(-1, 1, 1)
            loc = np.sum(prob * idx, axis=0)
            out_j = np.argmax(out_j, axis=0)
            loc[out_j == griding_num] = 0
            out_j = loc
        else:
            raise NotImplementedError
        name = names[j]

        line_save_path = os.path.join(output_path, name[:-3] + "lines.txt")
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(line_save_path, "w") as fp:
            for i in range(out_j.shape[1]):
                if np.sum(out_j[:, i] != 0) > 2:
                    for k in range(out_j.shape[0]):
                        if out_j[k, i] > 0:
                            fp.write(
                                "%d %d "
                                % (
                                    int(out_j[k, i] * col_sample_w * 1640 / 800) - 1,
                                    int(590 - k * 20) - 1,
                                )
                            )
                    fp.write("\n")


def run_test(
    net, data_root, exp_name, work_dir, griding_num, use_aux, distributed, batch_size=8
):
    # torch.backends.cudnn.benchmark = True
    output_path = os.path.join(work_dir, exp_name)
    if not os.path.exists(output_path) and is_main_process():
        os.mkdir(output_path)
    synchronize()
    loader = get_test_loader(batch_size, data_root, "CULane", distributed)
    device = next(net.parameters()).device
    # import pdb;pdb.set_trace()
    for i, data in enumerate(dist_tqdm(loader)):
        imgs, names = data
        imgs = imgs.to(device)
        with torch.no_grad():
            out = net(imgs)
        if len(out) == 2 and use_aux:
            out, seg_out = out

        generate_lines(
            out,
            imgs[0, 0].shape,
            names,
            output_path,
            griding_num,
            localization_type="rel",
            flip_updown=True,
        )


def generate_tusimple_lines(out, shape, griding_num, localization_type="rel"):

    out = out.data.cpu().numpy()
    out_loc = np.argmax(out, axis=0)

    if localization_type == "rel":
        prob = scipy.special.softmax(out[:-1, :, :], axis=0)
        idx = np.arange(griding_num)
        idx = idx.reshape(-1, 1, 1)

        loc = np.sum(prob * idx, axis=0)

        loc[out_loc == griding_num] = griding_num
        out_loc = loc
    lanes = []
    for i in range(out_loc.shape[1]):
        out_i = out_loc[:, i]
        lane = [
            int(round((loc + 0.5) * 1280.0 / (griding_num - 1)))
            if loc != griding_num
            else -2
            for loc in out_i
        ]
        lanes.append(lane)
    return lanes


def run_test_tusimple(
    net, data_root, work_dir, exp_name, griding_num, use_aux, distributed, batch_size=8
):
    output_path = os.path.join(work_dir, exp_name + ".%d.txt" % get_rank())
    fp = open(output_path, "w")
    loader = get_test_loader(batch_size, data_root, "Tusimple", distributed)
    device = next(net.parameters()).device
    for i, data in enumerate(dist_tqdm(loader)):
        imgs, names = data
        imgs = imgs.to(device)
        with torch.no_grad():
            out = net(imgs)
        if len(out) == 2 and use_aux:
            out = out[0]
        for i, name in enumerate(names):
            tmp_dict = {}
            tmp_dict["lanes"] = generate_tusimple_lines(
                out[i], imgs[0, 0].shape, griding_num
            )
            tmp_dict["h_samples"] = [
                160,
                170,
                180,
                190,
                200,
                210,
                220,
                230,
                240,
                250,
                260,
                270,
                280,
                290,
                300,
                310,
                320,
                330,
                340,
                350,
                360,
                370,
                380,
                390,
                400,
                410,
                420,
                430,
                440,
                450,
                460,
                470,
                480,
                490,
                500,
                510,
                520,
                530,
                540,
                550,
                560,
                570,
                580,
                590,
                600,
                610,
                620,
                630,
                640,
                650,
                660,
                670,
                680,
                690,
                700,
                710,
            ]
            tmp_dict["raw_file"] = name
            tmp_dict["run_time"] = 10
            json_str = json.dumps(tmp_dict)

            fp.write(json_str + "\n")
    fp.close()


def combine_tusimple_test(work_dir, exp_name):
    size = get_world_size()
    all_res = []
    for i in range(size):
        output_path = os.path.join(work_dir, exp_name + ".%d.txt" % i)
        with open(output_path, "r") as fp:
            res = fp.readlines()
        all_res.extend(res)
    names = set()
    all_res_no_dup = []
    for i, res in enumerate(all_res):
        pos = res.find("clips")
        name = res[pos:].split('"')[0]
        if name not in names:
            names.add(name)
            all_res_no_dup.append(res)

    output_path = os.path.join(work_dir, exp_name + ".txt")
    with open(output_path, "w") as fp:
        fp.writelines(all_res_no_dup)


def eval_lane(net, dataset, data_root, work_dir, griding_num, use_aux, distributed):
    net.eval()
    if dataset == "CULane":
        exp_name = "culane_eval_tmp"
        run_test(
            net,
            data_root,
            exp_name,
            work_dir,
            griding_num,
            use_aux,
            distributed,
        )
        synchronize()  # wait for all results
        if is_main_process():
            dist_print(
                "CULane Python-only mode: generated lane line prediction files without C++ benchmark evaluation."
            )
            dist_print(os.path.join(work_dir, exp_name))
        synchronize()

    elif dataset == "Tusimple":
        exp_name = "tusimple_eval_tmp"
        run_test_tusimple(
            net, data_root, work_dir, exp_name, griding_num, use_aux, distributed
        )
        synchronize()  # wait for all results
        if is_main_process():
            combine_tusimple_test(work_dir, exp_name)
            res = LaneEval.bench_one_submit(
                os.path.join(work_dir, exp_name + ".txt"),
                os.path.join(data_root, "test_label.json"),
            )
            res = json.loads(res)
            for r in res:
                dist_print(r["name"], r["value"])
        synchronize()
