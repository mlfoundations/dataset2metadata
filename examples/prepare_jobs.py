import os
from pathlib import Path

import fsspec
import yaml


if __name__ == '__main__':

    yml_template = yaml.safe_load(Path('custom/blip2.yml').read_text())
    num_tars_per_wds = 10
    num_wds_per_worker = 1
    jobs_dir_path = 'examples/jobs'


    fs, output_path = fsspec.core.url_to_fs(
        's3://s-laion/datanet_10M_pool-baselines/clip_threshold/clip_l14_f0.3/seed_0/'
    )

    shards = sorted(fs.glob(os.path.join(output_path, '**/*.tar')))
    shards = [f"pipe:aws s3 cp s3://{s} -" for s in shards]
    groups = [shards[i:i+num_tars_per_wds] for i in range(0, len(shards), num_tars_per_wds)]

    for i, g in enumerate(groups):
        yml_template['input_tars'] = g

        with open(os.path.join(jobs_dir_path, f'{i}.yml'), 'w') as f:
            for k in yml_template:
                f.write(f'{k}:')
                if not isinstance(yml_template[k], list):
                    f.write(f' {yml_template[k]}\n')
                else:
                    f.write('\n')
                    for v in yml_template[k]:
                        f.write(f'  - "{v}"\n')

    print(f'Saved {len(groups)} jobs to {jobs_dir_path}')
    print('Done.')


