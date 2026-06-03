import csv

import os

def dataLoad(fileName, year=2002):
    column = year - 2006
    listByYear = []
    for col in range(column ,column+3):
        f = open(fileName, 'rt')
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            listByYear.append(float(row[col]))

        f.close()
    return listByYear

def wtProducedPower(windSpeedList, turbinePowerNominal,cutInSpeed=2.5,
                    cutOffSpeed=30, ratedOutputSpeed=15):
    wPower = [0.0] * len(windSpeedList)
    for i in range(len(windSpeedList)):
        if windSpeedList[i] > cutInSpeed and windSpeedList[i] < cutOffSpeed:
            if windSpeedList[i] < ratedOutputSpeed:
                wPower[i] = round((turbinePowerNominal *
                                   ((windSpeedList[i]) - (cutInSpeed))) /
                                  ((ratedOutputSpeed) - (cutInSpeed)), 2)
            else:
                wPower[i] = round(turbinePowerNominal, 2)
    return wPower

def solarProducedPower(irradiance, pvEfficiency, numberPv):
    solarPower = [0] * len(irradiance)
    for i in range(len(irradiance)):
        solarPower[i] = round(
            ((pvEfficiency * irradiance[i] * numberPv)), 2)
    return solarPower

def renewablePowerProduced(numberPv,turbinePowerNominal):
    current_dir = os.getcwd()
    wind_file = os.path.join(current_dir, "./data/windspeedS.csv")
    solar_file=os.path.join(current_dir, "./data/irradianceS.csv")

    windListes = dataLoad(wind_file, 2007)
    windPowerLists = wtProducedPower(windListes, turbinePowerNominal)


    pvEfficiency = 0.2
    solarLists = dataLoad(solar_file, 2007)
    solarPowerLists = solarProducedPower(
        solarLists, pvEfficiency, numberPv)

    renewable = [a + b for a, b in zip(solarPowerLists , windPowerLists)]

    return renewable

class green_power():
    def __init__(self,greenWin,numberPv,turbinePowerNominal):
        self.greenPowerList=renewablePowerProduced(numberPv,turbinePowerNominal)
        self.greenWin=greenWin

    def getGreenPowerSlot(self,currentTime):
        greenPowerSlot=[]
        index=int(currentTime/3600)  #任务开始点的slot索引
        t=currentTime
        for i in range(index,index+self.greenWin):
            power=self.greenPowerList[i]
            lastTime=(i+1)*3600-t
            greenPowerSlot.append({'lastTime':lastTime,'power':power})
            t=(i+1)*3600
        return greenPowerSlot

    def getGreenUtili(self,power,start,end):
        usedGreen=0
        startIndex=int(start/3600)
        endIndex=int(end/3600)
        t=start
        for i in range(startIndex,endIndex+1):
            if i==endIndex:
                lastTime=end-t
            else:
                lastTime = (i + 1) * 3600 - t

            if power>=self.greenPowerList[i]:
                usedGreen+=self.greenPowerList[i]*lastTime
            else:
                usedGreen+=power*lastTime
            t=(i+1)*3600
        return usedGreen


    def getGreenPowerUtilization(self,powerSlot):
        totalEnergy=0
        usedGreen=0
        for i in range(len(powerSlot)-1):
            power=powerSlot[i]['power']
            start=powerSlot[i]['timeSlot']
            end=powerSlot[i+1]['timeSlot']
            lastTime=end-start
            totalEnergy+=lastTime*power
            inc_usedGreen=self.getGreenUtili(power,start,end)
            usedGreen+=inc_usedGreen

        if totalEnergy==0:
            return 1
        else:
            return usedGreen/totalEnergy
        # return -(totalEnergy-usedGreen)/3600000

    def getGreenUtiliEstimate(self,power,start,end,minIndex,maxIndex):
        usedGreen=0
        startIndex=int(start/3600)
        endIndex=int(end/3600)
        t=start
        for i in range(startIndex,endIndex+1):
            if i==endIndex:
                lastTime=end-t
            else:
                lastTime = (i + 1) * 3600 - t

            if i>maxIndex:
                greenIndex=minIndex+(i-minIndex)%(maxIndex-minIndex+1)
            else:
                greenIndex=i
            if power>=self.greenPowerList[greenIndex]:
                usedGreen+=self.greenPowerList[greenIndex]*lastTime
            else:
                usedGreen+=power*lastTime
            t=(i+1)*3600
        return usedGreen
