import numpy as np
import matplotlib.pyplot as plt
import json
import glob
import os
import datetime
import pandas as pd
import matplotlib
import seaborn as sns        
 
from scipy import interpolate

import math
from multiprocessing import Pool
from tqdm import tqdm

from asc2array_cls import asc2array_cls
from pre_processing_cls import pre_processing,rejectDat,rejectedByOutlier,re_sampling,getNearestValue
from makeFixation import fixation_detection

dt_now = datetime.datetime.now()
date = dt_now.strftime("%Y%m%d")
   
#%%

rootFolder = "./rawData/"
saveFolder = f"./data/{date}"

folderName = sorted(glob.glob("./rawData/*"))

log = glob.glob(f"{saveFolder}/s*")
log = [f.split('/')[-1][:3] for f in log]

folderList = [f for f in folderName if not f[-3:] in log]


#%%


cfg={
    "TIME_START":-1,
    'mmFlag':False,   
    'normFlag':True,
    'usedEye':1,
    'WID_FILTER':[],
    "RESAMPLING_RATE":250,
    # 'WID_FILTER':np.array([0.2,100]),
    's_trg':'Start_Experiment',
    'visualization':False,
    # 'visualization':True,
    "MS":False,
    "DOT_PITCH":0.278,   
    "VISUAL_DISTANCE":100,
    "acceptMSRange":4.5,
    "MIN_DURATION":0.1,
    "SCREEN_RES":[960*2, 540*2],
    "rejectFlag":[]
}

# cfg={   
#      "TIME_START":-1,
#      "TIME_START_RUN":-5,
#      "TIME_END":3,
#      "WID_ANALYSIS":3,
#      "WID_BP_ANALYSIS":2,
#      "usedEye":1, ### better eye
#        # "usedEye":0, ### both eyes
#      "WID_FILTER":[],
#      }

center = np.array(cfg['SCREEN_RES'])/2
    
#%%
def run(iSub):

    datHash={
            "PDR":[],
            "gazeX":[],
            "gazeY":[],
            "condition_frame_freq":[],
            "condition_frame_spl":[],
            "sub":[],
            "Run":[],
            "task":[],
            "numOfBlink":[],
            "numOfSaccade":[],
            "ampOfSaccade":[],
            "rejectFlag":[]
            }
    
    for iRun,ascFile in tqdm(enumerate(sorted(glob.glob(f"{iSub}/*.asc")))):

        cfg['usedEye'] = 1

        #% ------------------ data loading ------------------
        if iRun == 0:
            cfg['s_trg'] = 'Start_Experiment'
        else:
            cfg['s_trg'] = "Fixation"
        
        cfg["tmp_Run"] = iRun
        cfg['fName'] = ascFile
        
        t2a = asc2array_cls(cfg)
  
        #% ------------------ load json file (eye data) -------------------
        dat = t2a.dataExtraction(ascFile)
    
        #% ------------------ load json file (eye data) -------------------
        eyeData,events,initialTimeVal,fs = t2a.dataParse(dat)

        #% ------------------ guard: fully-lost eye channel ------------------
        # If one eye has (almost) no valid pupil samples, blinkInterp's
        # per-eye low-pass filter passes an empty vector to scipy.filtfilt,
        # which raises "input vector x must be greater than padlen, 15".
        # Copy the good eye into the dead channel and pin usedEye to the
        # good eye. (Hits s03_session6: left eye lost tracking all session.)
        pupil = eyeData["pupil"]
        for iEye in range(pupil.shape[0]):
            if np.count_nonzero(pupil[iEye]) <= 15:
                good = 1 - iEye
                print(f"  [guard] {os.path.basename(ascFile)}: eye{iEye} "
                      f"has no valid pupil data -> copied from eye{good}")
                pupil[iEye] = pupil[good].copy()
                orig = np.asarray(eyeData["pupilData_original"])
                orig[iEye] = orig[good].copy()
                eyeData["pupilData_original"] = orig
                cfg["usedEye"] = "R" if good == 1 else "L"

        ave,sigma = t2a.getAve(eyeData["pupil"])
        
        #% ------------------ load json file (eye data) -------------------
        eyeData = t2a.blinkInterp(eyeData)
        
        if cfg["normFlag"]:
            # pupilData = t2a.pupilNorm(eyeData["pupilData"], normDat[normDat["sub"]==iSub]["ave"][iSub], normDat[normDat["sub"]==iSub]["sigma"][iSub]).reshape(-1)
            pupilData = t2a.pupilNorm(eyeData["pupilData"], ave, sigma).reshape(-1)
        else:
            pupilData = np.mean(eyeData["pupilData"],axis=0)
            pupilData = pupilData.reshape(-1)
    
        if eyeData["usedEye"] == "L":
            iEyes=0
        else:
            iEyes=1
        
        zeroArray = np.repeat(eyeData["zeroArray"][iEyes].copy(),4)
        zeroArray = zeroArray[:len(pupilData)]
        
        st = np.argwhere( np.diff(zeroArray) == 1).reshape(-1)
        ed = np.argwhere( np.diff(zeroArray) == -1).reshape(-1)
        if len(ed) < len(st):
            ed = np.r_[ed,len(zeroArray)]
        
        for s,e in zip(st,ed):
            if e-s > fs*2:
                print("Oops! You closed your eyes for " + 
                      str(round((e-s)/fs,3)) + " from " + str(s) + " to "+ str(e))
                pupilData[s:e] = np.nan
        
        
        gazeX = np.array(eyeData["gazeX"])
        gazeY = np.array(eyeData["gazeY"])
        
        for mmName in ["gazeX","gazeY"]:
            for iEyes in np.arange(gazeX.shape[0]):
                exec(mmName+"[iEyes,"+mmName+"[iEyes,:]==0] = np.nan")
                
        gazeX = np.nanmean(gazeX,axis=0)
        gazeY = np.nanmean(gazeY,axis=0)
    
        
        coef = fs / 1000
        
        #% ------------------ event MSG ------------------
        events_onset = [[int(int(e[0])- initialTimeVal),e[1]] for e in events['MSG'] if e[1] == 'Start_Presentation']
        events_offset = [[int(int(e[0])- initialTimeVal),e[1]] for e in events['MSG'] if e[1] == 'End_Presentation']
        
        condition_frame_freq = [int(e[1][-1])for e in events['MSG'] if 'Condition_freq' in e[1]]
        condition_frame_spl = [e[1][-2:] for e in events['MSG'] if 'Condition_spl' in e[1]]
        condition_frame_spl = [int(e[-1]) if ":" in e else int(e) for e in condition_frame_spl]
        events_sound = [int(int(e[0])- initialTimeVal) for e in events['MSG'] if e[1] == 'PlaySound']
     
        
        #% ------------------ task response ------------------
        event_task = [[int(int(e[0])- initialTimeVal),e[1]] for e in events['MSG'] if e[1] == 'Task_Onset']
        event_task.append([int(int(events['MSG'][-1][0])-initialTimeVal),'dummy'])
        
        event_response = [[int(int(e[0])- initialTimeVal),e[1]] for e in events['MSG'] if e[1] == 'Task_response']
        
        time_response = []
        for i in np.arange(len(event_task)-1):
            ind_onset = event_task[i][0]
            ind_onset2 = event_task[i+1][0]
            tmp_res = []
            for res in event_response:
                if ind_onset < res[0] and ind_onset2 > res[0]:
                    time_response.append(res[0])
            if len(tmp_res) > 0:
                time_response.append(tmp_res)
        
        time_task = []
        for osnet,offset in zip(events_onset,events_offset):
            tmp = []
            for iTask in event_task:
                if osnet[0] < iTask[0] and offset[0] > iTask[0]:        
                    tmp.append(iTask[0])
            if len(tmp) > 0:
                time_task.append(tmp)
            else:
                time_task.append([])           
        
        for iTask in time_task:
            tmp0=[]
            for j in np.arange(len(iTask)):
                tmp1 = []
                for iRes in time_response:
                    if iTask[j] < iRes and iTask[j]+3000 > iRes:
                        tmp1.append(1)
                if len(tmp1)>0:
                    tmp0.append(tmp1[0])
                else:
                    tmp0.append([])
            datHash["task"].append(tmp0)      
     
        #% ------------------ data extraction ------------------    
        
        for i,(ind_onset,ind_offset) in enumerate(zip(events_onset,events_offset)):
            ind_s = int(ind_onset[0]*coef+cfg["TIME_START"]*fs)
            ind_e = int(ind_offset[0]*coef)
            
            datHash['PDR'].append(pupilData[ind_s:ind_e].tolist())
            datHash['gazeX'].append(gazeX[ind_s:ind_e].tolist())
            datHash['gazeY'].append(gazeY[ind_s:ind_e].tolist())
                
            datHash['sub'].append(iSub)
            datHash['Run'].append(int(iRun))
        
    
        datHash['condition_frame_freq'] = np.r_[datHash['condition_frame_freq'],
                                                np.array(condition_frame_freq).astype(int)]
        datHash['condition_frame_spl'] = np.r_[datHash['condition_frame_spl'],
                                               np.array(condition_frame_spl).astype(int)]    
        

    cfg["SAMPLING_RATE"] = fs
    datHash["cfg"] = cfg

    # resample the time-series to RESAMPLING_RATE (matches data/20230718)
    nSamples = round((20 * (1 / 0.9) + abs(cfg["TIME_START"])) * cfg["RESAMPLING_RATE"])
    for mmName in ["PDR", "gazeX", "gazeY"]:
        datHash[mmName] = re_sampling(datHash[mmName], nSamples)

    for mm in list(datHash.keys()):
        if not isinstance(datHash[mm], list):
            datHash[mm] = np.asarray(datHash[mm]).tolist()

    os.makedirs(saveFolder, exist_ok=True)

    with open(f"{saveFolder}/{iSub[-3:]}_trial.json", "w") as f:
        json.dump(datHash, f)

    
    return datHash



#%%    
if __name__ == '__main__':
 
    with Pool(4) as p:
        tmp_datHash = p.map(run, folderName)
    