import math
from PowerStruc import power_struc
from greenPower import green_power

class Machine:
    def __init__(self, id):
        self.id = id
        self.running_job_id = -1
        self.is_free = True
        self.job_history = []

    def taken_by_job(self, job_id):
        if self.is_free:
            self.running_job_id = job_id
            self.is_free = False
            self.job_history.append(job_id)
            return True
        else:
            return False

    def release(self):
        if self.is_free:
            return -1
        else:
            self.is_free = True
            self.running_job_id = -1
            return 1

    def reset(self):
        self.is_free = True
        self.running_job_id = -1
        self.job_history = []

    def __eq__(self, other):
        return self.id == other.id

    def __str__(self):
        return "M["+str(self.id)+"] "


class Cluster:
    def __init__(self, cluster_name, node_num, num_procs_per_node,processor_per_machine,idlePower,greenWin,numberPv,turbinePowerNominal):
        self.name = cluster_name
        self.total_node = node_num
        self.free_node = node_num
        self.used_node = 0
        self.num_procs_per_node = num_procs_per_node
        self.all_nodes = []
        self.statPower = idlePower*math.ceil(self.total_node / processor_per_machine)
        self.PowerStruc = power_struc(self.statPower)
        self.greenPower = green_power(greenWin,numberPv,turbinePowerNominal)
        self.green_win=greenWin
        for i in range(self.total_node):
            self.all_nodes.append(Machine(i))

    def feature(self):
        return [self.free_node]

    def can_allocated(self, job):
        if job.request_number_of_nodes != -1 and job.request_number_of_nodes > self.free_node:
            return False
        if job.request_number_of_nodes != -1 and job.request_number_of_nodes <= self.free_node:
            return True

        request_node = int(math.ceil(float(job.request_number_of_processors)/float(self.num_procs_per_node)))
        job.request_number_of_nodes = request_node
        if request_node > self.free_node:
            return False
        else:
            return True

    def allocate(self, job_id, request_num_procs):
        allocated_nodes = []
        request_node = int(math.ceil(float(request_num_procs) / float(self.num_procs_per_node)))

        if request_node > self.free_node:
            return []

        allocated = 0

        for m in self.all_nodes:
            if allocated == request_node:
                return allocated_nodes
            if m.taken_by_job(job_id):
                allocated += 1
                self.used_node += 1
                self.free_node -= 1
                allocated_nodes.append(m)

        if allocated == request_node:
            return allocated_nodes

        print ("Error in allocation, there are enough free resources but can not allocated!")
        return []

    def release(self, releases):
        self.used_node -= len(releases)
        self.free_node += len(releases)

        for m in releases:
            m.release()

    def is_idle(self):
        if self.used_node == 0:
            return True
        return False

    def reset(self):
        self.used_node = 0
        self.free_node = self.total_node
        self.PowerStruc = power_struc(self.statPower)
        for m in self.all_nodes:
            m.reset()

    def getGreenJobState(self,job,currentTime,currentSlot):
        start=currentTime
        end=start+job.request_time
        power=job.power
        jobEnergy=power*job.request_time
        powerSlot,beforeList = self.PowerStruc.getPre(start,end,power,currentSlot)
        totalEnergyBefore=0
        usedGreenBefore=0
        totalEnergyAfter=0
        usedGreenAfter=0
        minIndex=int(currentTime/3600)
        maxIndex=minIndex+self.green_win-1
        for i in range(len(powerSlot)-1):
            powerAfter=powerSlot[i]['power']
            startAfter=powerSlot[i]['timeSlot']
            endAfter=powerSlot[i+1]['timeSlot']
            lastTimeAfter=endAfter-startAfter
            consumeEnergyAfter=lastTimeAfter*powerAfter
            totalEnergyAfter+=consumeEnergyAfter
            inc_usedGreenAfter=self.greenPower.getGreenUtiliEstimate(powerAfter,startAfter,endAfter,minIndex,maxIndex)
            usedGreenAfter+=inc_usedGreenAfter

            powerBefore=beforeList[i]['power']
            startBefore=beforeList[i]['timeSlot']
            endBefore=beforeList[i+1]['timeSlot']
            lastTimeBefore=endBefore-startBefore
            consumeEnergyBefore=lastTimeBefore*powerBefore
            totalEnergyBefore+=consumeEnergyBefore
            inc_usedGreenBefore=self.greenPower.getGreenUtiliEstimate(powerBefore,startBefore,endBefore,minIndex,maxIndex)
            usedGreenBefore+=inc_usedGreenBefore

        BrownEnergyBefore=totalEnergyBefore-usedGreenBefore
        BrownEnergyAfter=totalEnergyAfter-usedGreenAfter

        jobBrownEnergy=BrownEnergyAfter-BrownEnergyBefore
        if jobBrownEnergy==0:
            return 0,0
        else:
            return 1,jobBrownEnergy/jobEnergy

    def backfill_check(self,running_jobs,job,current_time,backfill=1):
        if not self.can_allocated(job):
            return False
        if backfill==2:
            return True
        currentSlot=self.PowerStruc.getSlotFromRunning(running_jobs, current_time)
        slotList,beforeList=self.PowerStruc.getPre(current_time,
                                                        current_time + job.request_time,
                                                        job.power,currentSlot)
        totalEnergyBefore=0
        usedGreenBefore=0
        totalEnergyAfter=0
        usedGreenAfter=0
        minIndex=int(current_time/3600)
        maxIndex=minIndex+self.green_win-1

        for i in range(len(slotList) - 1):
            powerAfter = slotList[i]['power']
            startAfter = slotList[i]['timeSlot']
            endAfter = slotList[i + 1]['timeSlot']
            lastTimeAfter = endAfter - startAfter
            consumeEnergyAfter = lastTimeAfter * powerAfter
            totalEnergyAfter += consumeEnergyAfter
            inc_usedGreenAfter = self.greenPower.getGreenUtiliEstimate(powerAfter, startAfter, endAfter,minIndex,maxIndex)
            usedGreenAfter += inc_usedGreenAfter

            powerBefore = beforeList[i]['power']
            startBefore = beforeList[i]['timeSlot']
            endBefore = beforeList[i + 1]['timeSlot']
            lastTimeBefore = endBefore - startBefore
            consumeEnergyBefore = lastTimeBefore * powerBefore
            totalEnergyBefore += consumeEnergyBefore
            inc_usedGreenBefore = self.greenPower.getGreenUtiliEstimate(powerBefore, startBefore, endBefore,minIndex,maxIndex)
            usedGreenBefore += inc_usedGreenBefore

        BrownEnergyBefore = totalEnergyBefore - usedGreenBefore
        BrownEnergyAfter = totalEnergyAfter - usedGreenAfter

        jobBrownEnergy = BrownEnergyAfter - BrownEnergyBefore

        # return True #EASY

        if jobBrownEnergy < 50000:
            return True
        else:
            return False

    def LPTPN_check(self,running_jobs,job,current_time):

        currentSlot=self.PowerStruc.getSlotFromRunning(running_jobs, current_time)
        slotList,beforeList=self.PowerStruc.getPre(current_time,
                                                        current_time + job.request_time,
                                                        job.power,currentSlot)
        totalEnergyBefore=0
        usedGreenBefore=0
        totalEnergyAfter=0
        usedGreenAfter=0
        minIndex=int(current_time/3600)
        maxIndex=minIndex+self.green_win-1

        for i in range(len(slotList) - 1):
            powerAfter = slotList[i]['power']
            startAfter = slotList[i]['timeSlot']
            endAfter = slotList[i + 1]['timeSlot']
            lastTimeAfter = endAfter - startAfter
            consumeEnergyAfter = lastTimeAfter * powerAfter
            totalEnergyAfter += consumeEnergyAfter
            inc_usedGreenAfter = self.greenPower.getGreenUtiliEstimate(powerAfter, startAfter, endAfter,minIndex,maxIndex)
            usedGreenAfter += inc_usedGreenAfter

            powerBefore = beforeList[i]['power']
            startBefore = beforeList[i]['timeSlot']
            endBefore = beforeList[i + 1]['timeSlot']
            lastTimeBefore = endBefore - startBefore
            consumeEnergyBefore = lastTimeBefore * powerBefore
            totalEnergyBefore += consumeEnergyBefore
            inc_usedGreenBefore = self.greenPower.getGreenUtiliEstimate(powerBefore, startBefore, endBefore,minIndex,maxIndex)
            usedGreenBefore += inc_usedGreenBefore

        BrownEnergyBefore = totalEnergyBefore - usedGreenBefore
        BrownEnergyAfter = totalEnergyAfter - usedGreenAfter

        jobBrownEnergy = BrownEnergyAfter - BrownEnergyBefore

        if jobBrownEnergy < 50000:
            return True
        else:
            return False

    def backfill_check_ga(self,running_jobs,job,current_time,minGreen,backfill=1):
        if not self.can_allocated(job):
            return False
        if backfill==2:
            return True
        currentSlot=self.PowerStruc.getSlotFromRunning(running_jobs, current_time)
        slotList,beforeList=self.PowerStruc.getPre(current_time,
                                                        current_time + job.request_time,
                                                        job.power,currentSlot)
        totalEnergyBefore=0
        usedGreenBefore=0
        totalEnergyAfter=0
        usedGreenAfter=0
        minIndex=int(minGreen/3600)
        maxIndex=minIndex+self.green_win-1
        for i in range(len(slotList) - 1):
            powerAfter = slotList[i]['power']
            startAfter = slotList[i]['timeSlot']
            endAfter = slotList[i + 1]['timeSlot']
            lastTimeAfter = endAfter - startAfter
            consumeEnergyAfter = lastTimeAfter * powerAfter
            totalEnergyAfter += consumeEnergyAfter
            inc_usedGreenAfter = self.greenPower.getGreenUtiliEstimate(powerAfter, startAfter, endAfter,minIndex,maxIndex)
            usedGreenAfter += inc_usedGreenAfter

            powerBefore = beforeList[i]['power']
            startBefore = beforeList[i]['timeSlot']
            endBefore = beforeList[i + 1]['timeSlot']
            lastTimeBefore = endBefore - startBefore
            consumeEnergyBefore = lastTimeBefore * powerBefore
            totalEnergyBefore += consumeEnergyBefore
            inc_usedGreenBefore = self.greenPower.getGreenUtiliEstimate(powerBefore, startBefore, endBefore,minIndex,maxIndex)
            usedGreenBefore += inc_usedGreenBefore

        BrownEnergyBefore = totalEnergyBefore - usedGreenBefore
        BrownEnergyAfter = totalEnergyAfter - usedGreenAfter

        jobBrownEnergy = BrownEnergyAfter - BrownEnergyBefore


        if jobBrownEnergy < 50000:
            return True
        else:
            return False


class FakeList:
    def __init__(self, l):
        self.len = l
    def __len__(self):
        return self.len

class SimpleCluster:
    def __init__(self, cluster_name, node_num, num_procs_per_node):
        self.name = cluster_name
        self.total_node = node_num
        self.free_node = node_num
        self.used_node = 0
        self.num_procs_per_node = num_procs_per_node
        self.all_nodes = []

    def feature(self):
        return [self.free_node]

    def can_allocated(self, job):
        if job.request_number_of_nodes != -1:
            if job.request_number_of_nodes > self.free_node:
                return False
            else:
                return True

        request_node = int(math.ceil(float(job.request_number_of_processors)/float(self.num_procs_per_node)))
        job.request_number_of_nodes = request_node
        if request_node > self.free_node:
            return False
        else:
            return True

    def allocate(self, job_id, request_num_procs):
        allocated_nodes = FakeList(0)
        request_node = int(math.ceil(float(request_num_procs) / float(self.num_procs_per_node)))

        if request_node > self.free_node:
            return []

        allocated = request_node

        self.used_node += allocated
        self.free_node -= allocated
        allocated_nodes.len = allocated
        if allocated == request_node:
            return allocated_nodes

        print ("Error in allocation, there are enough free resources but can not allocated!")
        return []

    def release(self, releases):
        self.used_node -= len(releases)
        self.free_node += len(releases)


    def is_idle(self):
        if self.used_node == 0:
            return True
        return False

    def reset(self):
        self.used_node = 0
        self.free_node = self.total_node

