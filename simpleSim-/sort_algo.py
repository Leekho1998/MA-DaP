# 最先提交的容器优先
def first_submit_deploy(task_list):
    task_list.sort(key=lambda container: container.submit_time)
    return task_list

# 长作业优先
def long_duration_first_deploy(task_list):
    task_list.sort(key=lambda container: container.task_duration, reverse=True)
    return task_list

# 短作业优先调度
def short_duration_first_deploy(task_list):
    task_list.sort(key=lambda container: container.task_duration)
    return task_list

def good_selection(task_list):
    jobs = {}
    for sel in task_list:
        job_id = sel.task.job.job_id
        if job_id in jobs:
            jobs[job_id]["containers"].append(sel.id)
            jobs[job_id]["duration"] += sel.remaining_run_time
        else:
            jobs[job_id] = {
                "containers": [sel.id],
                "duration": sel.remaining_run_time
            }
    containers = sorted(jobs.items(), key=lambda item: item[1]['duration'], reverse=True)
    res = []
    for _, cs in containers:
        res += cs["containers"]
    return res

def sort_tasks(algorithm_name, task_list):
    if algorithm_name == 'FirstSubmit':
        return first_submit_deploy(task_list)
    elif algorithm_name == "LongFirst":
        return long_duration_first_deploy(task_list)
    elif algorithm_name == "ShortFirst":
        return short_duration_first_deploy(task_list)
    else:  # DRL and HeterogeneousNetworkAware
        return good_selection(task_list)