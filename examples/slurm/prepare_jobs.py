import argparse
import os
from pathlib import Path

import fsspec
import yaml
from tqdm import tqdm
from dataset2metadata.utils import download_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--yml_template", type=str, required=True, help="path to a templete yml"
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        required=True,
        help="path to a cache dir to save jobs that will be distributed across slurm cluster",
    )

    parser.add_argument(
        "--shard_dir",
        type=str,
        required=True,
        help="creates jobs out of all shards in this dir",
    )

    parser.add_argument(
        "--num_tars_per_wds",
        type=int,
        required=False,
        default=160,
        help="how many tars to group together for 1 job",
    )

    parser.add_argument(
        "--subdirs",
        action="store_true",
        required=False,
        default=False,
        help="parse subdirs within the shard_dir for parquets",
    )

    args = parser.parse_args()

    yml_template = yaml.safe_load(Path(args.yml_template).read_text())
    jobs_dir_path = args.cache_dir

    if not os.path.exists(jobs_dir_path):
        os.mkdir(jobs_dir_path)

    shards = None

    fs, output_path = fsspec.core.url_to_fs(args.shard_dir)

    if args.subdirs:
        all_dirs = [d for d in fs.ls(output_path) if "." not in d.split("/")[-1]]
        shards = []
        for dir in tqdm(all_dirs):
            shard_part = fs.glob(os.path.join(dir, "*.tar"))
            shards.extend(shard_part)
        shards = sorted(shards)
    else:
        shards = sorted(fs.glob(os.path.join(output_path, "*.tar")))

    if args.shard_dir.startswith("s3"):
        shards = [f"pipe:aws s3 cp s3://{s} -" for s in shards]

    groups = [
        shards[i : i + args.num_tars_per_wds]
        for i in range(0, len(shards), args.num_tars_per_wds)
    ]
    print(f"num shards: {len(shards)}")
    print(f"num jobs: {len(groups)}")

    for i, g in enumerate(groups):
        yml_template["input_tars"] = g

        with open(os.path.join(jobs_dir_path, f"{i}.yml"), "w") as f:
            for k in yml_template:
                f.write(f"{k}:")
                if not isinstance(yml_template[k], list):
                    f.write(f" {yml_template[k]}\n")
                else:
                    f.write("\n")
                    for v in yml_template[k]:
                        f.write(f'  - "{v}"\n')

    print(f"Saved {len(groups)} jobs to {jobs_dir_path}")

    download_all()
    print("Downloaded all default dataset2metadata checkpoints")

    print("Done.")
