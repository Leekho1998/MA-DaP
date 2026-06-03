class power_struc():
    def __init__(self, statPower):
        self.powerSlotLog=[]
        self.currentIndex=0
        self.statPower = statPower


    def reset(self):
        self.powerSlotLog=[]
        self.currentIndex=0

    def update(self,start,end,power):

        head_index = 0
        if len(self.powerSlotLog)==0:
            self.powerSlotLog.append({'timeSlot': start, 'power': power+self.statPower})
            self.powerSlotLog.append({'timeSlot': end, 'power': self.statPower})
            return
        for i in range(self.currentIndex,len(self.powerSlotLog)):
            if start>self.powerSlotLog[i]['timeSlot']:
                if i==len(self.powerSlotLog)-1:
                    self.powerSlotLog.append({'timeSlot': start, 'power': power+self.statPower})
                    self.powerSlotLog.append({'timeSlot': end, 'power': self.statPower})
                    return
                continue
            elif start==self.powerSlotLog[i]['timeSlot']:
                head_index=i
                self.powerSlotLog[i]['power'] += power
                if head_index == len(self.powerSlotLog) - 1:
                    self.powerSlotLog.append({'timeSlot': end, 'power': self.powerSlotLog[head_index]['power'] - power})
                    return
                break
            else:
                head_index=i
                beforeIndex=i-1
                newSloat={'timeSlot': start, 'power': power+self.powerSlotLog[beforeIndex]['power']}
                self.powerSlotLog.insert(head_index,newSloat)
                break

        for i in range(head_index+1,len(self.powerSlotLog)):
            if end>self.powerSlotLog[i]['timeSlot']:
                self.powerSlotLog[i]['power'] += power
                if i==len(self.powerSlotLog)-1:
                    self.powerSlotLog.append({'timeSlot': end, 'power': self.powerSlotLog[i]['power'] - power})
                    return
                continue
            elif end==self.powerSlotLog[i]['timeSlot']:
                return
            else:
                beforeIndex=i-1
                newSloat={'timeSlot': end, 'power': self.powerSlotLog[beforeIndex]['power']-power}
                self.powerSlotLog.insert(i,newSloat)
                return

    def updateCurrentTime(self,updateTime):
        for i in range(self.currentIndex,len(self.powerSlotLog)):
            if self.powerSlotLog[i]['timeSlot']==updateTime:
                self.currentIndex=i
                break
            if self.powerSlotLog[i]['timeSlot']>updateTime:
                if i>0:
                    self.currentIndex=i-1
                break
            if i==len(self.powerSlotLog)-1:
                self.currentIndex=len(self.powerSlotLog)-1
        return


    def getSlotFromRunning(self,running_jobs,currentTime):
        running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.request_time))
        currentSlot=[]
        if len(running_jobs)==0:
            return currentSlot
        lastPower=self.statPower
        lastJobPower =0
        for job in reversed(running_jobs):
            end=job.scheduled_time + job.request_time
            power=lastPower+lastJobPower
            lastPower=power
            currentSlot.append({'timeSlot': end, 'power': power})
            lastJobPower=job.power
        currentSlot.append({'timeSlot': currentTime, 'power': lastPower+lastJobPower})
        return currentSlot[::-1]

    def getPre(self,start,end,power,currentSlot):
        slotList=[]
        beforeList=[]
        head_index = 0
        if len(currentSlot)==0:
            slotList.append({'timeSlot': start, 'power': power+self.statPower})
            slotList.append({'timeSlot': end, 'power': self.statPower})
            beforeList.append({'timeSlot': start, 'power': self.statPower})
            beforeList.append({'timeSlot': end, 'power': self.statPower})
            return slotList,beforeList
        for i in range(len(currentSlot)):
            if start>currentSlot[i]['timeSlot']:
                if i==len(currentSlot)-1:
                    slotList.append({'timeSlot': start, 'power': power+self.statPower})
                    slotList.append({'timeSlot': end, 'power': self.statPower})
                    beforeList.append({'timeSlot': start, 'power': self.statPower})
                    beforeList.append({'timeSlot': end, 'power': self.statPower})
                    return slotList, beforeList
                continue
            elif start==currentSlot[i]['timeSlot']:
                head_index=i
                slotList.append({'timeSlot': start, 'power': currentSlot[i]['power'] + power})
                beforeList.append({'timeSlot': start, 'power': currentSlot[i]['power']})
                if head_index == len(currentSlot) - 1:
                    slotList.append({'timeSlot': end, 'power': currentSlot[head_index]['power']})
                    beforeList.append({'timeSlot': end, 'power': currentSlot[head_index]['power']})
                    return slotList,beforeList
                break
            else:
                head_index=i
                beforeIndex=i-1
                newSloat={'timeSlot': start, 'power': power+currentSlot[beforeIndex]['power']}
                slotList.append(newSloat)
                beforeList.append({'timeSlot': start, 'power': currentSlot[beforeIndex]['power']})
                break

        for i in range(head_index,len(currentSlot)):
            if end>currentSlot[i]['timeSlot']:
                slotList.append({'timeSlot': currentSlot[i]['timeSlot'], 'power':currentSlot[i]['power'] + power})
                beforeList.append({'timeSlot': currentSlot[i]['timeSlot'], 'power':currentSlot[i]['power']})
                if i==len(currentSlot)-1:
                    slotList.append({'timeSlot': end, 'power': currentSlot[i]['power']})
                    beforeList.append({'timeSlot': end, 'power': currentSlot[i]['power']})
                    return slotList,beforeList
                continue
            elif end==currentSlot[i]['timeSlot']:
                slotList.append({'timeSlot': end, 'power': currentSlot[i]['power']})
                beforeList.append({'timeSlot': end, 'power': currentSlot[i]['power']})
                return slotList,beforeList
            else:
                beforeIndex=i-1
                newSloat={'timeSlot': end, 'power': currentSlot[beforeIndex]['power']}
                slotList.append(newSloat)
                beforeList.append({'timeSlot': end, 'power': currentSlot[beforeIndex]['power']})
                return slotList,beforeList

