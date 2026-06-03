from job import Job, Workloads
from cluster import Cluster
import sys
import copy
import numpy as np
import scipy.signal
import gym
from gym import spaces
from gym.utils import seeding
import configparser

config = configparser.ConfigParser()
config.read('configFile/config.ini')


eta = float(config.get('GAS-MARL setting', 'eta'))
MAX_QUEUE_SIZE = int(config.get('GAS-MARL setting', 'MAX_QUEUE_SIZE'))
run_win = int(config.get('GAS-MARL setting', 'run_win'))
green_win = int(config.get('GAS-MARL setting', 'green_win'))
delayMaxJobNum = int(config.get('GAS-MARL setting', 'delayMaxJobNum'))
delayTimeList = eval(config.get('GAS-MARL setting', 'delayTimeList'))

turbinePowerNominal = float(config.get('general setting', 'turbinePowerNominal'))
numberPv = float(config.get('general setting', 'numberPv'))
processor_per_machine = int(config.get('general setting', 'processor_per_machine'))
idlePower = float(config.get('general setting', 'idlePower'))
MAX_perProcPower = float(config.get('general setting', 'MAX_perProcPower'))


MAX_POWER = 19000
MAX_GREEN = 19000
MAX_WAIT_TIME = 12 * 60 * 60
MAX_RUN_TIME = 12 * 60 * 60
JOB_FEATURES = 8
JOB_SEQUENCE_SIZE = MAX_QUEUE_SIZE
RUN_FEATURE = 4
GREEN_FEATURE = 2


action2_num= len(delayTimeList) + delayMaxJobNum + 1

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class HPCEnv(gym.Env):
    def __init__(self, backfill=False):  # do nothing and return. A workaround for passing parameters to the environment
        super(HPCEnv, self).__init__()
        print("Initialize Simple HPC Env")

        self.action_space = spaces.Discrete(MAX_QUEUE_SIZE)
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(JOB_FEATURES * MAX_QUEUE_SIZE,),
                                            dtype=np.float32)

        self.job_queue = []
        self.running_jobs = []
        self.visible_jobs = []
        self.pairs = []

        self.current_timestamp = 0
        self.start = 0
        self.next_arriving_job_idx = 0
        self.last_job_in_batch = 0
        self.num_job_in_batch = 0
        self.start_idx_last_reset = 0

        self.loads = None
        self.cluster = None

        self.scheduled_rl = {}

        self.backfill = backfill


    # @profile
    def my_init(self, workload_file='', sched_file=''):
        print("loading workloads from dataset:", workload_file)
        self.loads = Workloads(workload_file)
        self.cluster = Cluster("Cluster", self.loads.max_nodes, self.loads.max_procs / self.loads.max_nodes,processor_per_machine,idlePower,green_win,numberPv,turbinePowerNominal)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def f2_score(self, job):
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        request_time = job.request_time
        # run_time = job.run_time
        # f2: r^(1/2)*n + 25600 * log10(s)
        return (np.sqrt(request_time) * request_processors + 25600 * np.log10(submit_time))


    def fcfs_score(self, job):
        submit_time = job.submit_time
        return submit_time

    def lptpn_score(self, job):
        t = -job.power*job.request_time
        return t

    def backfill_score(self, job):
        t = job.power*job.request_time*job.request_number_of_processors  # Green-Backfilling

        return t

    # @profile
    def reset(self):
        self.cluster.reset()
        self.loads.reset()

        self.job_queue = []
        self.running_jobs = []
        self.visible_jobs = []
        self.pairs = []

        self.current_timestamp = 0
        self.start = 0
        self.next_arriving_job_idx = 0
        self.last_job_in_batch = 0
        self.num_job_in_batch = 0
        self.scheduled_rl = {}

        job_sequence_size = JOB_SEQUENCE_SIZE

        self.start = self.np_random.randint(job_sequence_size, (self.loads.size() - job_sequence_size - 1))

        self.start_idx_last_reset = self.start
        self.num_job_in_batch = job_sequence_size
        self.last_job_in_batch = self.start + self.num_job_in_batch
        self.current_timestamp = self.loads[self.start].submit_time
        self.job_queue.append(self.loads[self.start])
        self.next_arriving_job_idx = self.start + 1

        return self.build_observation()

    def reset_for_test(self, num, start):
        self.cluster.reset()
        self.loads.reset()

        self.job_queue = []
        self.running_jobs = []
        self.visible_jobs = []
        self.pairs = []

        self.current_timestamp = 0
        self.next_arriving_job_idx = 0
        self.last_job_in_batch = 0
        self.num_job_in_batch = 0
        self.scheduled_rl = {}

        job_sequence_size = num
        self.start = start

        self.start_idx_last_reset = self.start
        self.num_job_in_batch = job_sequence_size
        self.last_job_in_batch = self.start + self.num_job_in_batch
        self.current_timestamp = self.loads[self.start].submit_time
        self.job_queue.append(self.loads[self.start])
        self.next_arriving_job_idx = self.start + 1

    def skip_for_resources_greedy(self, job, scheduled_logs):
        # note that this function is only called when current job can not be scheduled.
        assert not self.cluster.can_allocated(job)

        while not self.cluster.can_allocated(job):
            # schedule nothing, just move forward to next timestamp. It should just add a new job or finish a running job
            assert self.running_jobs
            self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
            next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
            next_resource_release_machines = self.running_jobs[0].allocated_machines

            if self.next_arriving_job_idx < self.last_job_in_batch and self.loads[
                self.next_arriving_job_idx].submit_time <= next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                self.cluster.release(next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job.

    def skip_for_resources_LPTPN(self, job, scheduled_logs):
        # note that this function is only called when current job can not be scheduled.
        # assert not self.cluster.can_allocated(job)

        while not self.cluster.can_allocated(job) or not self.cluster.LPTPN_check(self.running_jobs, job, self.current_timestamp) :
            # schedule nothing, just move forward to next timestamp. It should just add a new job or finish a running job
            # assert self.running_jobs
            if len(self.running_jobs)==0:
                break
            self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
            next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
            next_resource_release_machines = self.running_jobs[0].allocated_machines

            nextGreenChange = self.current_timestamp + 1
            # nextGreenChange = self.current_timestamp + 1
            if self.next_arriving_job_idx < self.last_job_in_batch \
                    and self.loads[self.next_arriving_job_idx].submit_time <= min(next_resource_release_time,nextGreenChange):
                self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
            elif nextGreenChange < next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp, nextGreenChange)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                self.cluster.release(next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job

    # @profile
    def moveforward_for_resources_backfill_greedy(self, job, scheduled_logs):
        # note that this function is only called when current job can not be scheduled.
        assert not self.cluster.can_allocated(job)

        earliest_start_time = self.current_timestamp
        # sort all running jobs by estimated finish time
        self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.request_time))
        free_processors = self.cluster.free_node * self.cluster.num_procs_per_node
        for running_job in self.running_jobs:
            free_processors += len(running_job.allocated_machines) * self.cluster.num_procs_per_node
            earliest_start_time = (running_job.scheduled_time + running_job.request_time)
            if free_processors >= job.request_number_of_processors:
                break

        while not self.cluster.can_allocated(job):

            # try to backfill as many jobs as possible.
            if self.backfill == 1:
                self.job_queue.sort(key=lambda _j: self.backfill_score(_j))
            else:
                self.job_queue.sort(key=lambda _j: self.fcfs_score(_j))
            job_queue_iter_copy = list(self.job_queue)

            self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.request_time))
            free_processors = self.cluster.free_node * self.cluster.num_procs_per_node
            temp_est=earliest_start_time
            for running_job in self.running_jobs:
                free_processors += len(running_job.allocated_machines) * self.cluster.num_procs_per_node
                temp_est = (running_job.scheduled_time + running_job.request_time)
                if free_processors >= job.request_number_of_processors:
                    break
            earliest_start_time=max(earliest_start_time,temp_est)
            for _j in job_queue_iter_copy:
                if _j!=job and (self.current_timestamp + _j.request_time) < earliest_start_time:
                    if self.cluster.backfill_check(self.running_jobs, _j, self.current_timestamp, self.backfill):
                        # we should be OK to schedule the job now
                        assert _j.scheduled_time == -1  # this job should never be scheduled before.
                        _j.scheduled_time = self.current_timestamp
                        _j.allocated_machines = self.cluster.allocate(_j.job_id, _j.request_number_of_processors)
                        self.cluster.PowerStruc.update(_j.scheduled_time,
                                                       _j.scheduled_time + _j.run_time,
                                                       _j.power)
                        self.running_jobs.append(_j)
                        score = self.job_score(_j)  # calculated reward
                        scheduled_logs[_j.job_id] = score
                        self.job_queue.remove(_j)  # remove the job from job queue

            # move to the next timestamp
            assert self.running_jobs
            self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
            next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
            next_resource_release_machines = self.running_jobs[0].allocated_machines

            nextGreenChange = ((self.current_timestamp // 3600) + 1) * 3600
            if self.next_arriving_job_idx < self.last_job_in_batch \
                    and self.loads[self.next_arriving_job_idx].submit_time <= min(next_resource_release_time,nextGreenChange):
                self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
            elif nextGreenChange < next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp, nextGreenChange)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                self.cluster.release(next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job

        self.job_queue.sort(key=lambda _j: self.fcfs_score(_j))

    def post_process_score(self, scheduled_logs):
        for i in scheduled_logs:
            scheduled_logs[i] /= self.num_job_in_batch

    # @profile
    def schedule_curr_sequence_reset(self, score_fn):
        # schedule the sequence of jobs using heuristic algorithm.
        scheduled_logs = {}
        while True:
            self.job_queue.sort(key=lambda j: score_fn(j))
            job_for_scheduling = self.job_queue[0]

            # if selected job needs more resources, skip scheduling and try again after adding new jobs or releasing some resources
            if not self.cluster.can_allocated(job_for_scheduling):
                if self.backfill:
                    self.moveforward_for_resources_backfill_greedy(job_for_scheduling, scheduled_logs)
                else:
                    self.skip_for_resources_greedy(job_for_scheduling, scheduled_logs)

            assert job_for_scheduling.scheduled_time == -1  # this job should never be scheduled before.
            job_for_scheduling.scheduled_time = self.current_timestamp
            job_for_scheduling.allocated_machines = self.cluster.allocate(job_for_scheduling.job_id,
                                                                          job_for_scheduling.request_number_of_processors)
            self.cluster.PowerStruc.update(job_for_scheduling.scheduled_time,
                                           job_for_scheduling.scheduled_time + job_for_scheduling.run_time,
                                           job_for_scheduling.power)
            self.running_jobs.append(job_for_scheduling)
            score = self.job_score(job_for_scheduling)  # calculated reward
            scheduled_logs[job_for_scheduling.job_id] = score
            self.job_queue.remove(job_for_scheduling)

            not_empty = self.moveforward_for_job()
            if not not_empty:
                break
        self.post_process_score(scheduled_logs)
        greenRwd = self.cluster.greenPower.getGreenPowerUtilization(self.cluster.PowerStruc.powerSlotLog)

        self.cluster.reset()
        self.loads.reset()
        self.job_queue = []
        self.running_jobs = []
        self.visible_jobs = []
        self.pairs = []
        self.current_timestamp = self.loads[self.start].submit_time
        self.job_queue.append(self.loads[self.start])
        self.last_job_in_batch = self.start + self.num_job_in_batch
        self.next_arriving_job_idx = self.start + 1

        return scheduled_logs,greenRwd

    def build_observation(self):
        vector = np.zeros((MAX_QUEUE_SIZE + run_win + green_win) * JOB_FEATURES, dtype=float)
        self.job_queue.sort(key=lambda job: self.fcfs_score(job))

        self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.request_time))
        currentSlot=self.cluster.PowerStruc.getSlotFromRunning(self.running_jobs,self.current_timestamp)
        self.pairs = [
                         [
                             job,
                             min(float(self.current_timestamp - job.submit_time) / float(MAX_WAIT_TIME), 1.0 - 1e-5),
                             min(float(job.request_time) / float(self.loads.max_exec_time), 1.0 - 1e-5),
                             min(float(job.request_number_of_processors) / float(self.loads.max_procs), 1.0 - 1e-5),
                             min(float(job.power) / float(MAX_POWER), 1.0 - 1e-5),
                             min(float(job.power / job.request_number_of_processors) / float(MAX_perProcPower),
                                 1.0 - 1e-5),
                             *self.cluster.getGreenJobState(job, self.current_timestamp, currentSlot),
                             1.0 - 1e-5 if self.cluster.can_allocated(job) else 1e-5
                         ]
                         for i, job in enumerate(self.job_queue)
                         if i < MAX_QUEUE_SIZE
                     ] + [
                         [None, 0, 1, 1, 1, 1, 1, 1, 0]
                         for _ in range(MAX_QUEUE_SIZE - len(self.job_queue))
                     ]

        vector[:MAX_QUEUE_SIZE * JOB_FEATURES] = [item for pair in self.pairs[:MAX_QUEUE_SIZE] for item in pair[1:]]

        running_job = [
                          [
                              min(float(temp_job.request_number_of_processors) / float(self.loads.max_procs),
                                  1.0 - 1e-5),
                              min(float(temp_job.power) / float(MAX_POWER), 1.0 - 1e-5),
                              min(float(temp_job.power / temp_job.request_number_of_processors) / float(
                                  MAX_perProcPower), 1.0 - 1e-5),
                              min(float(temp_job.scheduled_time + temp_job.request_time - self.current_timestamp) / float(
                                  self.loads.max_exec_time), 1.0 - 1e-5),
                              0,0,0,0
                          ]
                          for i, temp_job in enumerate(self.running_jobs[:run_win])
                          if i < run_win
                      ] + [
                          [0, 0, 0, 0, 0,0,0,0]
                          for _ in range(run_win - len(self.running_jobs))
                      ]
        vector[MAX_QUEUE_SIZE * JOB_FEATURES:(MAX_QUEUE_SIZE + run_win) * JOB_FEATURES] = [
            job_feature for job in running_job for job_feature in job
        ]



        green = self.cluster.greenPower.getGreenPowerSlot(self.current_timestamp)
        green_slot = [
            [
                min(float(greenPower['lastTime']) / float(self.loads.max_exec_time), 1.0 - 1e-5),
                min(float(greenPower['power']) / float(MAX_GREEN), 1.0 - 1e-5),
                0,0,0,0,0,0
            ]
            for greenPower in green
        ]

        start_index = MAX_QUEUE_SIZE + run_win
        end_index = MAX_QUEUE_SIZE + run_win + green_win
        vector[start_index * JOB_FEATURES:end_index * JOB_FEATURES] = [item for slot in green_slot[
                                                                                        start_index - MAX_QUEUE_SIZE - run_win:end_index - MAX_QUEUE_SIZE - run_win]
                                                                       for item in slot]

        return vector

    # @profile
    def moveforward_for_resources_backfill(self, job):
        # note that this function is only called when current job can not be scheduled.
        assert not self.cluster.can_allocated(job)

        earliest_start_time = self.current_timestamp
        # sort all running jobs by estimated finish time
        self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.request_time))
        free_processors = self.cluster.free_node * self.cluster.num_procs_per_node
        for running_job in self.running_jobs:
            free_processors += len(running_job.allocated_machines) * self.cluster.num_procs_per_node
            earliest_start_time = (running_job.scheduled_time + running_job.request_time)
            if free_processors >= job.request_number_of_processors:
                break

        while not self.cluster.can_allocated(job):
            # try to backfill as many jobs as possible
            if self.backfill==1:
                self.job_queue.sort(key=lambda _j: self.backfill_score(_j))
            else:
                self.job_queue.sort(key=lambda _j: self.fcfs_score(_j))
            job_queue_iter_copy = list(self.job_queue)

            for _j in job_queue_iter_copy:
                if  _j!=job and (self.current_timestamp + _j.request_time) < earliest_start_time and self.cluster.backfill_check(self.running_jobs, _j, self.current_timestamp, self.backfill):
                    # we should be OK to schedule the job now
                    assert _j.scheduled_time == -1  # this job should never be scheduled before.
                    _j.scheduled_time = self.current_timestamp
                    _j.allocated_machines = self.cluster.allocate(_j.job_id, _j.request_number_of_processors)
                    self.cluster.PowerStruc.update(_j.scheduled_time,
                                                   _j.scheduled_time + _j.run_time,
                                                   _j.power)
                    self.running_jobs.append(_j)
                    score = self.job_score(_j)  # calculated reward
                    self.scheduled_rl[_j.job_id] = score
                    self.job_queue.remove(_j)  # remove the job from job queue

            # move to the next timestamp
            assert self.running_jobs
            self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
            next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
            next_resource_release_machines = self.running_jobs[0].allocated_machines

            nextGreenChange = ((self.current_timestamp // 3600) + 1) * 3600
            if self.next_arriving_job_idx < self.last_job_in_batch \
                    and self.loads[self.next_arriving_job_idx].submit_time <= min(next_resource_release_time,nextGreenChange):
                self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
            elif nextGreenChange < next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp, nextGreenChange)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                self.cluster.release(next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job
        self.job_queue.sort(key=lambda _j: self.fcfs_score(_j))

    def skip_for_resources(self, job):
        # note that this function is only called when current job can not be scheduled.
        assert not self.cluster.can_allocated(job)

        while not self.cluster.can_allocated(job):
            # schedule nothing, just move forward to next timestamp. It should just add a new job or finish a running job
            assert self.running_jobs
            self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
            next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
            next_resource_release_machines = self.running_jobs[0].allocated_machines

            if self.next_arriving_job_idx < self.last_job_in_batch and self.loads[
                self.next_arriving_job_idx].submit_time <= next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                self.cluster.release(next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job.

    # @profile
    def moveforward_for_job(self):
        if self.job_queue:
            return True

        # if we need to add job, but can not add any more, return False indicating the job_queue is for sure empty now.
        if self.next_arriving_job_idx >= self.last_job_in_batch:
            assert not self.job_queue
            return False

        # move forward to add jobs into job queue.
        while not self.job_queue:
            if not self.running_jobs:  # there are no running jobs
                next_resource_release_time = sys.maxsize  # always add jobs if no resource can be released.
                next_resource_release_machines = []
            else:
                self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
                next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
                next_resource_release_machines = self.running_jobs[0].allocated_machines

            if self.loads[self.next_arriving_job_idx].submit_time <= next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
                return True  # job added
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                self.cluster.release(next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job.

    def job_score(self, job_for_scheduling):

        _tmp = max(1.0, (float(
            job_for_scheduling.scheduled_time - job_for_scheduling.submit_time + job_for_scheduling.run_time)
                         /
                         max(job_for_scheduling.run_time, 10)))
        return _tmp

    def has_only_one_job(self):
        if len(self.job_queue) == 1:
            return True
        else:
            return False

    def schedule(self, job_for_scheduling):
        # make sure we move forward and release needed resources
        if not self.cluster.can_allocated(job_for_scheduling):
            self.skip_for_resources(job_for_scheduling)

        # we should be OK to schedule the job now
        assert job_for_scheduling.scheduled_time == -1  # this job should never be scheduled before.
        job_for_scheduling.scheduled_time = self.current_timestamp
        job_for_scheduling.allocated_machines = self.cluster.allocate(job_for_scheduling.job_id,
                                                                      job_for_scheduling.request_number_of_processors)
        self.cluster.PowerStruc.update(job_for_scheduling.scheduled_time,
                                       job_for_scheduling.scheduled_time + job_for_scheduling.run_time,
                                       job_for_scheduling.power)
        self.running_jobs.append(job_for_scheduling)
        score = self.job_score(job_for_scheduling)  # calculated reward
        self.scheduled_rl[job_for_scheduling.job_id] = score
        self.job_queue.remove(job_for_scheduling)  # remove the job from job queue

        # after scheduling, check if job queue is empty, try to add jobs.
        not_empty = self.moveforward_for_job()

        if not_empty:
            # job_queue is not empty
            return False
        else:
            # job_queue is empty and can not add new jobs as we reach the end of the sequence
            return True

    def valid(self, a):
        action = a[0]
        return self.pairs[action][0]

    def skip1(self,a2):
        self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.request_time))
        release_index=a2-1
        release_time = (self.running_jobs[release_index].scheduled_time + self.running_jobs[release_index].request_time)
        skipTime = min(release_time,3600+self.current_timestamp)
        next_time_after_skip = skipTime

        next_resource_release_time = sys.maxsize  # always add jobs if no resource can be released.
        next_resource_release_machines = []
        next_job_sumbitTime = sys.maxsize
        if self.running_jobs:  # there are running jobs
            self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
            next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
            next_resource_release_machines = self.running_jobs[0].allocated_machines

        if self.next_arriving_job_idx < self.last_job_in_batch:
            next_job_sumbitTime=self.loads[self.next_arriving_job_idx].submit_time

        while True:
            if next_time_after_skip < min(next_job_sumbitTime,next_resource_release_time):
                self.current_timestamp = max(self.current_timestamp,next_time_after_skip)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                return
            if next_job_sumbitTime <= next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp,
                                             self.loads[self.next_arriving_job_idx].submit_time)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1

                if self.next_arriving_job_idx < self.last_job_in_batch:
                    next_job_sumbitTime = self.loads[self.next_arriving_job_idx].submit_time
                else:
                    next_job_sumbitTime=sys.maxsize
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                self.cluster.release(next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job.
                if len(self.running_jobs)>0:
                    next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
                    next_resource_release_machines = self.running_jobs[0].allocated_machines
                else:
                    next_resource_release_time = sys.maxsize

    def skip2(self, skipTime):
        next_time_after_skip = self.current_timestamp + skipTime

        next_resource_release_time = sys.maxsize  # always add jobs if no resource can be released.
        next_resource_release_machines = []

        next_job_sumbitTime = sys.maxsize
        if self.running_jobs:  # there are running jobs
            self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
            next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
            next_resource_release_machines = self.running_jobs[0].allocated_machines

        if self.next_arriving_job_idx < self.last_job_in_batch:
            next_job_sumbitTime=self.loads[self.next_arriving_job_idx].submit_time

        while True:
            if next_time_after_skip < min(next_job_sumbitTime,next_resource_release_time):
                self.current_timestamp = max(self.current_timestamp,next_time_after_skip)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                return
            if next_job_sumbitTime <= next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp,
                                             self.loads[self.next_arriving_job_idx].submit_time)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1

                if self.next_arriving_job_idx < self.last_job_in_batch:
                    next_job_sumbitTime = self.loads[self.next_arriving_job_idx].submit_time
                else:
                    next_job_sumbitTime=sys.maxsize
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                self.cluster.release(next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job.
                if len(self.running_jobs)>0:
                    next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
                    next_resource_release_machines = self.running_jobs[0].allocated_machines
                else:
                    next_resource_release_time = sys.maxsize

    def moveforward_green_backfilling_delay_action1(self, job, a):
        self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.request_time))
        release_index=a-1
        release_time = (self.running_jobs[release_index].scheduled_time + self.running_jobs[release_index].request_time)
        skipTime = min(release_time,3600+self.current_timestamp)

        earliest_start_time = self.current_timestamp
        # sort all running jobs by estimated finish time
        free_processors = self.cluster.free_node * self.cluster.num_procs_per_node
        if free_processors < job.request_number_of_processors:
            for running_job in self.running_jobs:
                free_processors += len(running_job.allocated_machines) * self.cluster.num_procs_per_node
                earliest_start_time = (running_job.scheduled_time + running_job.request_time)
                if free_processors >= job.request_number_of_processors:
                    break
        earliest_start_time = max(earliest_start_time, skipTime)

        while not self.cluster.can_allocated(job) or self.current_timestamp<skipTime:

            self.job_queue.sort(key=lambda _j: self.backfill_score(_j))
            job_queue_iter_copy = list(self.job_queue)

            for _j in job_queue_iter_copy:
                if _j!=job and (self.current_timestamp + _j.request_time) < earliest_start_time and self.cluster.backfill_check(self.running_jobs, _j, self.current_timestamp, self.backfill):
                    # we should be OK to schedule the job now
                    assert _j.scheduled_time == -1  # this job should never be scheduled before.
                    _j.scheduled_time = self.current_timestamp
                    _j.allocated_machines = self.cluster.allocate(_j.job_id, _j.request_number_of_processors)
                    self.cluster.PowerStruc.update(_j.scheduled_time,
                                                   _j.scheduled_time + _j.run_time,
                                                   _j.power)
                    self.running_jobs.append(_j)
                    score = self.job_score(_j)  # calculated reward
                    self.scheduled_rl[_j.job_id] = score
                    self.job_queue.remove(_j)  # remove the job from job queue

            # move to the next timestamp
            assert self.running_jobs
            self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
            next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
            next_resource_release_machines = self.running_jobs[0].allocated_machines

            nextGreenChange = ((self.current_timestamp // 3600) + 1) * 3600
            if self.next_arriving_job_idx < self.last_job_in_batch \
                    and self.loads[self.next_arriving_job_idx].submit_time <= min(next_resource_release_time,nextGreenChange):
                self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
            elif nextGreenChange < next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp, nextGreenChange)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                self.cluster.release(next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job
        self.job_queue.sort(key=lambda _j: self.fcfs_score(_j))

    def moveforward_green_backfilling_delay_action2(self, job, ToskipTime):
        self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.request_time))
        skipTime = ToskipTime+self.current_timestamp

        earliest_start_time = self.current_timestamp
        # sort all running jobs by estimated finish time
        free_processors = self.cluster.free_node * self.cluster.num_procs_per_node
        if free_processors < job.request_number_of_processors:
            for running_job in self.running_jobs:
                free_processors += len(running_job.allocated_machines) * self.cluster.num_procs_per_node
                earliest_start_time = (running_job.scheduled_time + running_job.request_time)
                if free_processors >= job.request_number_of_processors:
                    break
        earliest_start_time = max(earliest_start_time, skipTime)

        while not self.cluster.can_allocated(job) or self.current_timestamp<skipTime:
            self.job_queue.sort(key=lambda _j: self.backfill_score(_j))
            job_queue_iter_copy = list(self.job_queue)

            for _j in job_queue_iter_copy:
                if  _j!=job and (self.current_timestamp + _j.request_time) < earliest_start_time and self.cluster.backfill_check(self.running_jobs, _j, self.current_timestamp, self.backfill):
                    # we should be OK to schedule the job now
                    assert _j.scheduled_time == -1  # this job should never be scheduled before.
                    _j.scheduled_time = self.current_timestamp
                    _j.allocated_machines = self.cluster.allocate(_j.job_id, _j.request_number_of_processors)
                    self.cluster.PowerStruc.update(_j.scheduled_time,
                                                   _j.scheduled_time + _j.run_time,
                                                   _j.power)
                    self.running_jobs.append(_j)
                    score = self.job_score(_j)  # calculated reward
                    self.scheduled_rl[_j.job_id] = score
                    self.job_queue.remove(_j)  # remove the job from job queue

            next_resource_release_time = sys.maxsize  # always add jobs if no resource can be released.
            next_resource_release_machines = []
            if self.running_jobs:  # there are running jobs
                self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
                next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
                next_resource_release_machines = self.running_jobs[0].allocated_machines

            next_job_sumbitTime = sys.maxsize
            if self.next_arriving_job_idx < self.last_job_in_batch:
                next_job_sumbitTime = self.loads[self.next_arriving_job_idx].submit_time

            nextGreenChange = ((self.current_timestamp // 3600) + 1) * 3600
            if skipTime > self.current_timestamp and skipTime < min(next_job_sumbitTime,next_resource_release_time,nextGreenChange):
                self.current_timestamp = max(self.current_timestamp,skipTime)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
            elif next_job_sumbitTime <= min(next_resource_release_time,nextGreenChange):
                self.current_timestamp = max(self.current_timestamp,
                                             self.loads[self.next_arriving_job_idx].submit_time)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
            elif nextGreenChange < next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp, nextGreenChange)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                self.cluster.release(next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job.
        self.job_queue.sort(key=lambda _j: self.fcfs_score(_j))


    def schedule_backfill(self, job_for_scheduling,a2):
        if a2==0:
            if not self.cluster.can_allocated(job_for_scheduling):
                self.moveforward_for_resources_backfill(job_for_scheduling)
        else:
            if a2 >0 and a2<=delayMaxJobNum:
                self.moveforward_green_backfilling_delay_action1(job_for_scheduling, a2)
            elif a2 > delayMaxJobNum:
                ToskipTime = delayTimeList[a2 - delayMaxJobNum - 1]
                self.moveforward_green_backfilling_delay_action2(job_for_scheduling, ToskipTime)

        # we should be OK to schedule the job now
        assert job_for_scheduling.scheduled_time == -1  # this job should never be scheduled before.
        job_for_scheduling.scheduled_time = self.current_timestamp
        job_for_scheduling.allocated_machines = self.cluster.allocate(job_for_scheduling.job_id,
                                                                      job_for_scheduling.request_number_of_processors)
        self.cluster.PowerStruc.update(job_for_scheduling.scheduled_time,
                                       job_for_scheduling.scheduled_time + job_for_scheduling.run_time,
                                       job_for_scheduling.power)
        self.running_jobs.append(job_for_scheduling)
        score = self.job_score(job_for_scheduling)  # calculated reward
        self.scheduled_rl[job_for_scheduling.job_id] = score
        self.job_queue.remove(job_for_scheduling)  # remove the job from job queue

        # after scheduling, check if job queue is empty, try to add jobs.
        not_empty = self.moveforward_for_job()

        if not_empty:
            # job_queue is not empty
            return False
        else:
            # job_queue is empty and can not add new jobs as we reach the end of the sequence
            return True

    def schedule_backfill_EASY(self, job_for_scheduling,a2):
        if a2 > 0 and a2 <= delayMaxJobNum:
            self.skip1(a2)
        elif a2 > delayMaxJobNum:
            skipTime = delayTimeList[a2 - delayMaxJobNum - 1]
            self.skip2(skipTime)
        if not self.cluster.can_allocated(job_for_scheduling):
            self.moveforward_for_resources_backfill(job_for_scheduling)

        # we should be OK to schedule the job now
        assert job_for_scheduling.scheduled_time == -1  # this job should never be scheduled before.
        job_for_scheduling.scheduled_time = self.current_timestamp
        job_for_scheduling.allocated_machines = self.cluster.allocate(job_for_scheduling.job_id,
                                                                      job_for_scheduling.request_number_of_processors)
        self.cluster.PowerStruc.update(job_for_scheduling.scheduled_time,
                                       job_for_scheduling.scheduled_time + job_for_scheduling.run_time,
                                       job_for_scheduling.power)
        self.running_jobs.append(job_for_scheduling)
        score = self.job_score(job_for_scheduling)  # calculated reward
        self.scheduled_rl[job_for_scheduling.job_id] = score
        self.job_queue.remove(job_for_scheduling)  # remove the job from job queue

        # after scheduling, check if job queue is empty, try to add jobs.
        not_empty = self.moveforward_for_job()

        if not_empty:
            # job_queue is not empty
            return False
        else:
            # job_queue is empty and can not add new jobs as we reach the end of the sequence
            return True


    def schedule_LPTPN_sequence_reset(self):
        # schedule the sequence of jobs using LPTPN algorithm.
        scheduled_logs = {}
        while True:
            self.job_queue.sort(key=lambda j: self.lptpn_score(j))
            flag=1
            for job_for_scheduling in self.job_queue:
                if self.cluster.can_allocated(job_for_scheduling) and self.cluster.LPTPN_check(self.running_jobs,
                                                                                                      job_for_scheduling,
                                                                                                      self.current_timestamp):
                    assert job_for_scheduling.scheduled_time == -1  # this job should never be scheduled before.
                    job_for_scheduling.scheduled_time = self.current_timestamp
                    job_for_scheduling.allocated_machines = self.cluster.allocate(job_for_scheduling.job_id,
                                                                                  job_for_scheduling.request_number_of_processors)
                    self.cluster.PowerStruc.update(job_for_scheduling.scheduled_time,
                                                   job_for_scheduling.scheduled_time + job_for_scheduling.run_time,
                                                   job_for_scheduling.power)
                    self.running_jobs.append(job_for_scheduling)
                    score = self.job_score(job_for_scheduling)  # calculated reward
                    scheduled_logs[job_for_scheduling.job_id] = score
                    self.job_queue.remove(job_for_scheduling)
                    flag=0
            not_empty = self.moveforward_for_job()
            if not not_empty:
                break

            if flag:
                if len(self.running_jobs)==0:
                    job_for_scheduling=self.job_queue[0]
                    assert job_for_scheduling.scheduled_time == -1  # this job should never be scheduled before.
                    job_for_scheduling.scheduled_time = self.current_timestamp
                    job_for_scheduling.allocated_machines = self.cluster.allocate(job_for_scheduling.job_id,
                                                                                  job_for_scheduling.request_number_of_processors)
                    self.cluster.PowerStruc.update(job_for_scheduling.scheduled_time,
                                                   job_for_scheduling.scheduled_time + job_for_scheduling.run_time,
                                                   job_for_scheduling.power)
                    self.running_jobs.append(job_for_scheduling)
                    score = self.job_score(job_for_scheduling)  # calculated reward
                    scheduled_logs[job_for_scheduling.job_id] = score
                    self.job_queue.remove(job_for_scheduling)
                    if not not_empty:
                        break
                else:
                    self.skip_for_resources_LPTPN(job_for_scheduling, scheduled_logs)


        self.post_process_score(scheduled_logs)
        greenRwd = self.cluster.greenPower.getGreenPowerUtilization(self.cluster.PowerStruc.powerSlotLog)

        self.cluster.reset()
        self.loads.reset()
        self.job_queue = []
        self.running_jobs = []
        self.visible_jobs = []
        self.pairs = []
        self.current_timestamp = self.loads[self.start].submit_time
        self.job_queue.append(self.loads[self.start])
        self.last_job_in_batch = self.start + self.num_job_in_batch
        self.next_arriving_job_idx = self.start + 1

        return scheduled_logs,greenRwd

    def step(self, a1,a2):
        job_for_scheduling = self.pairs[a1][0]
        if self.backfill==1:
            done=self.schedule_backfill(job_for_scheduling,a2)
        if self.backfill==2:
            done=self.schedule_backfill_EASY(job_for_scheduling,a2)
        elif self.backfill==0:
            if a2 >0 and a2<=delayMaxJobNum:
                self.skip1(a2)
            elif a2 > delayMaxJobNum:
                skipTime = delayTimeList[a2 - delayMaxJobNum - 1]
                self.skip2(skipTime)
            done = self.schedule(job_for_scheduling)

        if not done:
            obs = self.build_observation()
            return [obs, 0, False, 0, 0, 0,len(self.running_jobs),0]
        else:
            self.post_process_score(self.scheduled_rl)
            rl_total = sum(self.scheduled_rl.values())

            rwd = -rl_total
            greenRwd = self.cluster.greenPower.getGreenPowerUtilization(self.cluster.PowerStruc.powerSlotLog)

            return [None, rwd, True, 0, 0, 0,len(self.running_jobs),greenRwd]

    def step_for_ga(self, a1,a2):
        job_for_scheduling = self.job_queue[a1]
        if self.backfill==1:
            done=self.schedule_backfill(job_for_scheduling,a2)
        if self.backfill==2:
            done=self.schedule_backfill_EASY(job_for_scheduling,a2)
        elif self.backfill==0:
            if a2 >0 and a2<=delayMaxJobNum:
                self.skip1(a2)
            elif a2 > delayMaxJobNum:
                skipTime = delayTimeList[a2 - delayMaxJobNum - 1]
                self.skip2(skipTime)
            done = self.schedule(job_for_scheduling)

        if not done:
            obs = self.build_observation()
            return [obs, 0, False,len(self.running_jobs),0]
        else:
            self.post_process_score(self.scheduled_rl)
            rl_total = sum(self.scheduled_rl.values())
            rwd = -rl_total
            greenRwd = self.cluster.greenPower.getGreenPowerUtilization(self.cluster.PowerStruc.powerSlotLog)
            return [None, rwd, True,len(self.running_jobs),greenRwd]


    def job_score1(self, job_for_scheduling):
       # 用于计算适应度
       _tmp = max(1.0, (float(
           job_for_scheduling.scheduled_time - job_for_scheduling.submit_time + job_for_scheduling.request_time)
                        /
                        max(job_for_scheduling.request_time, 10)))
       return _tmp

    def skip_for_resources_ga(self, job, power_stru,free_processors,runningJobs,CurrrentTimestamp):

        while job.request_number_of_processors > free_processors:
            # schedule nothing, just move forward to next timestamp. It should just add a new job or finish a running job
            runningJobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.request_time))
            next_resource_release_time = (runningJobs[0].scheduled_time + runningJobs[0].request_time)

            CurrrentTimestamp=max(CurrrentTimestamp, next_resource_release_time)
            power_stru.updateCurrentTime(CurrrentTimestamp)
            free_processors+= runningJobs[0].request_number_of_processors
            runningJobs.pop(0)  # remove the first running job.
        return free_processors,CurrrentTimestamp

    def moveforward_for_resources_backfill_ga(self, job, power_stru,jobs_list,free_processors,runningJobs,CurrentTimestamp,scheduled_logs):

        earliest_start_time = CurrentTimestamp
        runningJobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.request_time))
        free_copy=copy.deepcopy(free_processors)

        for running_job in runningJobs:
            free_copy += runningJobs[0].request_number_of_processors
            earliest_start_time = (running_job.scheduled_time + running_job.request_time)
            if free_copy >= job.request_number_of_processors:
                break

        backfillJobList=copy.deepcopy(jobs_list)
        if self.backfill == 1:
            backfillJobList.sort(key=lambda _j: self.backfill_score(_j))
        else:
            backfillJobList.sort(key=lambda _j: self.fcfs_score(_j))

        while job.request_number_of_processors > free_processors:
            job_queue_iter_copy = list(backfillJobList)

            runningJobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.request_time))
            for _j in job_queue_iter_copy:
                if _j!=job and (CurrentTimestamp + _j.request_time) < earliest_start_time:
                    if _j.request_number_of_processors <= free_processors and \
                            self.cluster.backfill_check_ga(runningJobs, _j, CurrentTimestamp , self.current_timestamp, self.backfill):
                        # we should be OK to schedule the job now
                        assert _j.scheduled_time == -1  # this job should never be scheduled before.
                        _j.scheduled_time = CurrentTimestamp
                        free_processors-= _j.request_number_of_processors
                        power_stru.update(_j.scheduled_time,
                                                       _j.scheduled_time + _j.request_time,
                                                       _j.power)
                        runningJobs.append(_j)
                        score = self.job_score(_j)  # calculated reward
                        scheduled_logs[_j.job_id] = score
                        jobs_list.remove(_j)  # remove the job from job queue
                        backfillJobList.remove(_j)

            # move to the next timestamp
            assert runningJobs
            runningJobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
            next_resource_release_time = (runningJobs[0].scheduled_time + runningJobs[0].run_time)

            nextGreenChange = ((CurrentTimestamp // 3600) + 1) * 3600
            if nextGreenChange <  next_resource_release_time:
                CurrentTimestamp = max(CurrentTimestamp, next_resource_release_time)
                power_stru.updateCurrentTime(CurrentTimestamp)
            else:
                CurrentTimestamp = max(CurrentTimestamp, next_resource_release_time)
                power_stru.updateCurrentTime(CurrentTimestamp)
                free_processors += runningJobs[0].request_number_of_processors
                runningJobs.pop(0)  # remove the first running job

        return free_processors,CurrentTimestamp

    def getfitness(self,solution,Temp_power):
        free_processors=self.cluster.free_node * self.cluster.num_procs_per_node
        runningJobs=copy.deepcopy(self.running_jobs)
        CurrentTimestamp =copy.deepcopy(self.current_timestamp)

        jobs_list = copy.deepcopy([self.job_queue[solution[i]] for i in range(len(self.job_queue))])
        scheduled_logs={}
        while len(jobs_list)>0:
            job_for_scheduling = jobs_list[0]

            if job_for_scheduling.request_number_of_processors > free_processors:
                if self.backfill:
                    free_processors,CurrentTimestamp=self.moveforward_for_resources_backfill_ga(job_for_scheduling, Temp_power,jobs_list,free_processors,runningJobs,CurrentTimestamp,scheduled_logs)
                else:
                    free_processors,CurrentTimestamp=self.skip_for_resources_ga(job_for_scheduling, Temp_power,free_processors,runningJobs,CurrentTimestamp)

            assert job_for_scheduling.scheduled_time == -1  # this job should never be scheduled before.
            job_for_scheduling.scheduled_time = CurrentTimestamp
            free_processors-=job_for_scheduling.request_number_of_processors
            Temp_power.update(job_for_scheduling.scheduled_time,
                                           job_for_scheduling.scheduled_time + job_for_scheduling.request_time,
                                           job_for_scheduling.power)
            runningJobs.append(job_for_scheduling)
            score = self.job_score1(job_for_scheduling)  # calculated reward
            scheduled_logs[job_for_scheduling.job_id] = score
            jobs_list.remove(job_for_scheduling)

        self.post_process_score(scheduled_logs)
        rl_total = sum(scheduled_logs.values())
        rwd1 = -rl_total
        greenRwd=self.cluster.greenPower.getGreenPowerUtilization(Temp_power.powerSlotLog)

        return rwd1,greenRwd

