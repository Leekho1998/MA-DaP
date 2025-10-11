import numpy as np

def deal_with_affi(affi):
    return (affi + 1) ** 2 if affi > 0 else 0   # 亲和性

def deal_with_resource(host_resource, container_resource):
    res = host_resource-container_resource
    res[res<0] = 0
    return res   # 亲和性

def features_normalize_func(x):  # x是17维  原本是6维：cpu mem  taskcpu taskmem taskduration taskinstances
    x = np.array(x)
    max_vals =np.array([5,10,1,5,10, 8000,128,800,1,1,1]) + 1e-8
    bias = np.zeros(len(x))  # np.array([0, 0, 0, 0, 0, 0, 0])
    y = (np.array(x) - bias) / max_vals
    #print(f"normolized: {y}")
    return y

def scale_action(action, high, low):
    action = np.clip(action, -1, 1)
    weight = (high - low) / 2
    bias = (high + low) / 2
    action_ = action * weight + bias

    return action_