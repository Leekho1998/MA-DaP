import argparse
import ast
import math
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = [
    "job_name",
    "submit_time",
    "task_name",
    "instance_num",
    "task_duration",
    "plan_cpu",
    "plan_mem",
    "plan_gpu",
    "gpu_type",
    "communicate_count",
    "communicate_size",
    "decline",
    "preq",
    "children",
    "qualified",
]

TIME_SLOT_US = 600_000_000
HOST_CPU = 8000
HOST_MEM = 128
HOST_GPU = 800
GPU_TYPES = np.array(["MISC", "V100", "T4", "P100", "A100"], dtype=object)


def parse_mapping(value):
    if pd.isna(value) or value == "":
        return {}
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, SyntaxError):
        return {}


def round_to(value, base, lower, upper):
    rounded = int(math.ceil(float(value) / base) * base)
    return int(np.clip(rounded, lower, upper))


def first_valid_time(df):
    for column in ("start_time", "time"):
        values = pd.to_numeric(df[column], errors="coerce")
        values = values[values.notna() & (values >= 0)]
        if not values.empty:
            return int(values.min())
    return 0


def scan_submit_range(input_path, usecols, limit, chunksize):
    seen = 0
    min_time = None
    max_time = None
    for chunk in pd.read_csv(input_path, usecols=usecols, chunksize=chunksize):
        if limit is not None and seen >= limit:
            break
        chunk["start_time"] = pd.to_numeric(chunk["start_time"], errors="coerce")
        chunk["time"] = pd.to_numeric(chunk["time"], errors="coerce")
        submit_source = chunk["start_time"].where(chunk["start_time"].notna(), chunk["time"])
        submit_source = submit_source[submit_source.notna() & (submit_source >= 0)]
        if submit_source.empty:
            continue
        if limit is not None:
            submit_source = submit_source.iloc[: max(0, limit - seen)]
        seen += len(submit_source)
        cur_min = int(submit_source.min())
        cur_max = int(submit_source.max())
        min_time = cur_min if min_time is None else min(min_time, cur_min)
        max_time = cur_max if max_time is None else max(max_time, cur_max)
    return min_time or 0, max_time or 0


def generate_google_dataset(input_path, output_path, limit=2340, chunksize=50_000, seed=42, time_slots=432):
    rng = np.random.default_rng(seed)
    input_path = Path(input_path)
    output_path = Path(output_path)

    written = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    usecols = [
        "time",
        "collection_id",
        "scheduling_class",
        "collection_type",
        "priority",
        "instance_index",
        "resource_request",
        "start_time",
        "end_time",
        "average_usage",
        "maximum_usage",
        "assigned_memory",
        "cluster",
        "event",
        "failed",
    ]

    if limit is not None and limit <= 0:
        limit = None
    base_time, max_time = scan_submit_range(input_path, ["time", "start_time"], limit, chunksize)
    time_span = max(max_time - base_time, 1)

    for chunk in pd.read_csv(input_path, usecols=usecols, chunksize=chunksize):
        if limit is not None and written >= limit:
            break

        chunk = chunk[chunk["collection_id"].notna()].copy()
        chunk["start_time"] = pd.to_numeric(chunk["start_time"], errors="coerce")
        chunk["end_time"] = pd.to_numeric(chunk["end_time"], errors="coerce")
        chunk["time"] = pd.to_numeric(chunk["time"], errors="coerce")
        chunk["priority"] = pd.to_numeric(chunk["priority"], errors="coerce").fillna(0)
        chunk["scheduling_class"] = pd.to_numeric(chunk["scheduling_class"], errors="coerce").fillna(0)
        chunk["instance_index"] = pd.to_numeric(chunk["instance_index"], errors="coerce").fillna(0)

        submit_source = chunk["start_time"].where(chunk["start_time"].notna(), chunk["time"])
        valid_submit = submit_source.notna() & (submit_source >= 0)
        chunk = chunk[valid_submit].copy()
        submit_source = submit_source[valid_submit]
        if chunk.empty:
            continue

        duration_us = chunk["end_time"] - chunk["start_time"]
        duration_steps = np.ceil(duration_us / TIME_SLOT_US)
        fallback_duration = rng.lognormal(mean=2.2, sigma=0.45, size=len(chunk))
        duration_steps = np.where(duration_steps.notna() & (duration_steps > 0), duration_steps, fallback_duration)
        duration_steps = np.clip(duration_steps.astype(int), 6, 144)

        reqs = chunk["resource_request"].map(parse_mapping)
        avg_usage = chunk["average_usage"].map(parse_mapping)
        max_usage = chunk["maximum_usage"].map(parse_mapping)

        cpus = []
        mems = []
        gpu_probs = []
        comm_counts = []
        comm_sizes = []
        declines = []

        for idx, (_, row) in enumerate(chunk.iterrows()):
            req = reqs.iloc[idx]
            avg = avg_usage.iloc[idx]
            max_use = max_usage.iloc[idx]

            cpu_frac = req.get("cpus")
            if cpu_frac is None:
                cpu_frac = max_use.get("cpus", avg.get("cpus", 0.02))
            mem_frac = req.get("memory")
            if mem_frac is None:
                mem_frac = row["assigned_memory"] if pd.notna(row["assigned_memory"]) else max_use.get("memory", avg.get("memory", 0.01))

            cpus.append(round_to(max(float(cpu_frac), 0.001) * HOST_CPU, 100, 100, HOST_CPU))
            mems.append(round_to(max(float(mem_frac), 0.001) * HOST_MEM, 1, 1, HOST_MEM))

            gpu_prob = 0.03 + 0.04 * min(float(row["scheduling_class"]), 3.0) + 0.0005 * min(float(row["priority"]), 1000.0)
            gpu_probs.append(float(np.clip(gpu_prob, 0.03, 0.65)))

            if duration_steps[idx] <= 6:
                resource_intensity = min(float(cpu_frac or 0.02) + float(mem_frac or 0.01), 1.0)
                duration_steps[idx] = int(np.clip(rng.normal(10 + 18 * resource_intensity, 3), 6, 42))
            duration = duration_steps[idx]
            comm_count = int(np.clip(rng.poisson(1.0 + min(duration, 30) / 20.0), 0, min(6, duration)))
            comm_counts.append(comm_count)
            usage_hint = max(float(cpu_frac or 0.02), float(mem_frac or 0.01), 0.001)
            comm_sizes.append(int(np.clip(rng.lognormal(4.8 + usage_hint, 0.7) * max(comm_count, 1), 16, 8192)))

            priority = float(row["priority"])
            sched = float(row["scheduling_class"])
            if priority >= 360 or sched >= 3:
                decline = rng.integers(6, 25)
            elif priority >= 200:
                decline = rng.integers(12, 49)
            else:
                decline = rng.integers(24, 97)
            declines.append(int(decline))

        gpu_mask = rng.random(len(chunk)) < np.array(gpu_probs)
        gpu_amount = np.where(gpu_mask, rng.choice(np.arange(50, 251, 50), size=len(chunk)), 0)
        gpu_type = np.where(gpu_mask, rng.choice(GPU_TYPES[1:], size=len(chunk)), "MISC")

        if time_slots is not None and time_slots > 1:
            submit_time = np.floor((submit_source - base_time) / time_span * (time_slots - 1)).astype(int)
        else:
            submit_time = np.floor((submit_source - base_time) / TIME_SLOT_US).astype(int)
        submit_time = np.clip(submit_time, 0, None)

        out = pd.DataFrame(
            {
                "job_name": "google_" + chunk["collection_id"].astype("int64").astype(str),
                "submit_time": submit_time,
                "task_name": "instance_" + chunk["collection_id"].astype("int64").astype(str) + "_" + chunk["instance_index"].astype("int64").astype(str),
                "instance_num": 1,
                "task_duration": duration_steps,
                "plan_cpu": cpus,
                "plan_mem": mems,
                "plan_gpu": gpu_amount,
                "gpu_type": gpu_type,
                "communicate_count": comm_counts,
                "communicate_size": comm_sizes,
                "decline": declines,
                "preq": "",
                "children": "",
                "qualified": "",
            },
            columns=OUTPUT_COLUMNS,
        )

        if limit is not None:
            out = out.iloc[: max(0, limit - written)]

        out.to_csv(output_path, mode="a", index=False, header=written == 0)
        written += len(out)

    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./borg_traces_data.csv")
    parser.add_argument("--output", default="./google.csv")
    parser.add_argument("--limit", type=int, default=2340)
    parser.add_argument("--chunksize", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-slots", type=int, default=432)
    args = parser.parse_args()

    rows = generate_google_dataset(args.input, args.output, args.limit, args.chunksize, args.seed, args.time_slots)
    print(f"saved {args.output}, rows={rows}")


if __name__ == "__main__":
    main()
