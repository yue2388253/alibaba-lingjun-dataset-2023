import ast
from datetime import datetime
import pandas as pd


def parse_worker():
    start_time = datetime.strptime('2023/07/15 00:00', '%Y/%m/%d %H:%M')

    df = pd.read_csv('../data/worker.csv')
    stat_dict = {
        'initial_count': len(df),
    }

    # Count missing time entries before dropna
    missing_created = df['gmt_created'].isna()
    missing_finished = df['gmt_pod_finished'].isna()
    stat_dict['missing_time'] = (missing_created | missing_finished).sum()

    # Remove entries with any missing 'gmt_created' or 'gmt_pod_finished'
    df.dropna(subset=['gmt_created', 'gmt_pod_finished'], how='any', inplace=True)

    df['gmt_created'] = df['gmt_created'].apply(
        lambda x: (datetime.strptime(x, '%Y/%m/%d %H:%M') - start_time).total_seconds() / 86400
    )
    df['gmt_pod_finished'] = df['gmt_pod_finished'].apply(
        lambda x: (datetime.strptime(x, '%Y/%m/%d %H:%M') - start_time).total_seconds() / 86400
    )
    df['duration'] = df['gmt_pod_finished'] - df['gmt_created']

    # Check for invalid duration
    invalid_duration_df = df[df['duration'] <= 0]
    stat_dict['invalid_duration'] = len(invalid_duration_df)
    df = df[df['duration'] > 0]

    # Drop jobs whose host_ip is not in the topo.csv
    host_ip_set = set(pd.read_csv('../data/topo.csv')['ip'])
    invalid_ip_df = df[~df['host_ip'].isin(host_ip_set)]
    stat_dict['invalid_host_ip'] = len(invalid_ip_df)
    df = df[df['host_ip'].isin(host_ip_set)]

    # Check for missing RES
    missing_res_df = df[df['RES'].isna()]
    stat_dict['missing_res'] = len(missing_res_df['job_name'].unique())
    df = df.dropna(subset=['RES'])

    df['RES'] = df['RES'].apply(ast.literal_eval)
    df['num_gpus'] = df['RES'].apply(lambda x: int(x['nvidia.com/gpu']) if 'nvidia.com/gpu' in x else None)

    return df, stat_dict


df_worker_valid = parse_worker()
df_worker_GPU = df_worker_valid[df_worker_valid['num_gpus'].notna()]


def parse_job():
    global df_worker_GPU
    jobs = df_worker_GPU['job_name'].unique()
    groups = df_worker_GPU.groupby('job_name')

    list_start_time = []
    list_end_time = []
    list_num_gpus = []
    for job in jobs:
        start_time = groups.get_group(job)['gmt_created'].min()
        end_time = groups.get_group(job)['gmt_pod_finished'].max()
        num_gpus = groups.get_group(job)['num_gpus'].sum()

        list_start_time.append(start_time)
        list_end_time.append(end_time)
        list_num_gpus.append(num_gpus)

    df_jobs = pd.DataFrame({
        "job_name": jobs,
        "start_time": list_start_time,
        "end_time": list_end_time,
        "num_gpus": list_num_gpus
    })

    df_jobs['duration'] = df_jobs['end_time'] - df_jobs['start_time']
    return df_jobs


df_worker_jobs = parse_job()

