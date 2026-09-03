"""
Input:    
dat      - list of asc file transfered from EDF Converter 
              provided from SR-research(https://www.sr-support.com/thread-23.html)
cfg      - dict of parameters for analysis

Output:
eyeData   - dict which includes pupil, gaze and micro-saccades
events    - list of triggers in asc file
initialTimeVal - recordig start timing
fs        - sampling rate of the recording

Example:
    fileName = glob.glob(os.path.join('/xxx.asc'))
    f = open(os.path.join(str(fileName[0])))
    dat=[]
    for line in f.readlines():
        dat.append(line.split())
    f.close()
    cfg={'usedEye':2,
        'WID_FILTER':[],
        'mmFlag':False,
        'normFlag':True,
        's_trg':'SYNCTIME',
        'visualization':False
        }
    eyeData,events,initialTimeVal,fs = asc2array(dat, cfg)
    
    pupilData = eyeData['pupilData']
    gazeX = eyeData['gazeX']
    gazeY = eyeData['gazeY']
    mSaccade = eyeData['mSaccade']
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import datetime
import pandas as pd
import glob
import seaborn as sns
#from makeEyemetrics import makeMicroSaccade,draw_heatmap
import json
from zeroInterp import zeroInterp,zeroInterp2,zeroInterp_oneshot
from pre_processing import re_sampling
from au2mm import au2mm,getmm
from band_pass_filter import butter_bandpass_filter,lowpass_filter
from pre_processing_cls import getNearestValue
import zipfile
import statsmodels.formula.api as smf
# import zmq, msgpack
 
sns.set(font_scale=0.8)
sns.set_style("whitegrid")
sns.set_palette("Set2")

#%%
class asc2array_cls:    
    def __init__(self, cfg):
        self.cfg = cfg
    
        if len(cfg['s_trg']) > 0:
            self.s_trg = cfg['s_trg']
        else:
            self.s_trg = 'Start_Experiment'

        if self.cfg["visualization"]:
            self.figSize = [5,4]

        self.mmName = ['Left','Right']
        
    def dataExtraction(self,fileName,encode="utf-8"):
        
        fName, ext = os.path.splitext(fileName)
        dat=[]
        if ext==".json":
            # dat=pd.read_json(fileName)
            # dat = json.loads(fileName)
            # with open(fileName) as f:
            #     dat.append(f.readlines())
                
            with open(fileName, 'r') as f:
                dat = json.load(f)

        else:
            f = open(fileName,encoding = encode)
            
            for line in f.readlines():
                dat.append(line.split())
                
            f.close()
        
        return dat
    
    def dataParse(self, dat, EyeTracker = "Eyelink"):
        
        print('---------------------------------')
        print(f"Analysing...{self.cfg['fName']}")
  
        eyeData= {'Right':{
                    "timestamp":[],
                    "gazeX":[],
                    "gazeY":[],
                    "pupil":[],
                    },
                'Left':{
                    "timestamp":[],
                    "gazeX":[],
                    "gazeY":[],
                    "pupil":[],
                    }}
        
        if EyeTracker == "Eyelink":
            print('Eyelink is used.')
            events = {'SFIX':[],'EFIX':[],'SSACC':[],'ESACC':[],'SBLINK':[],'EBLINK':[],'MSG':[]}
            
            msg_type = ['SFIX','EFIX','SSACC','ESACC','SBLINK','EBLINK','MSG']
            
            for line in dat:
                if '!MODE' in line:
                    eyeRecording = line[-1]
                    break
           
            start = False
            for line in dat:
                if start:
                    # aaaa
                    if len(line) > 3:
                            
                        if line[0].isdecimal():
                            eyeData['Left']["timestamp"].append(float(line[0]))
                                
                            if line[1].replace('.','',1).isdigit():
                                eyeData['Left']["gazeX"].append(float(line[1]))
                            else:
                                eyeData['Left']["gazeX"].append(0)

                            if line[2].replace('.','',1).isdigit():
                                eyeData['Left']["gazeY"].append(float(line[2]))
                            else:
                                eyeData['Left']["gazeY"].append(0)
                            
                            if line[3].replace('.','',1).isdigit():
                                eyeData['Left']["pupil"].append(float(line[3]))
                            else:
                                eyeData['Left']["pupil"].append(0)
                                                       
                            if eyeRecording == "LR":
                                eyeData['Right']["timestamp"].append(float(line[0]))
                                
                                if line[4].replace('.','',1).isdigit() :
                                    eyeData['Right']["gazeX"].append(float(line[4]))
                                else:
                                    eyeData['Right']["gazeX"].append(0)
                                    
                                if line[5].replace('.','',1).isdigit() :
                                    eyeData['Right']["gazeY"].append(float(line[5]))
                                else:
                                    eyeData['Right']["gazeY"].append(0)
                                    
                                if line[6].replace('.','',1).isdigit() :
                                    eyeData['Right']["pupil"].append(float(line[6]))
                                else:
                                    eyeData['Right']["pupil"].append(0)
                                    
                            else:
                                eyeData['Right']["timestamp"].append(float(line[0]))
                                eyeData['Right']["gazeX"].append(float(line[1]))
                                eyeData['Right']["gazeY"].append(float(line[2]))
                                eyeData['Right']["pupil"].append(float(line[3]))
                    
                    if len(line)==0:
                        continue
                    
                    for m in msg_type:
                        if line[0] == m:
                              events[m].append(line[1:])
                else:
                    if 'PUPIL' in line:
                        self.Method = line[1]
                        
                    elif 'RATE' in line:
                        for l in line:
                            if l.replace('.','',1).isdigit():
                                # fs = float(l)
                                self.fs = float(l)
                                break
                            
                    if self.s_trg in line:
                        start = True
                        self.initialTimeVal = int(line[1])
                        if self.s_trg != 'Start_Experiment':
                            events['MSG'].append(line[1:])
                    
                    # s_trg = "START_EXPERIMENT"
                    # if s_trg in line:
                    #     start = True
                    #     initialTimeVal = int(line[1])
                    #     if s_trg != 'Start_Experiment':
                    #         events['MSG'].append(line[1:])
            
            print(f"Measurement method is {self.Method}")
                   
            # else:
            #     start = False
            #     for line in dat:
            #         if start:
            #             if len(line) > 3:
            #                 if line[0].isdecimal() and line[1].replace('.','',1).isdigit() :
            #                     eyeData['Left'].append([float(line[0]),
            #                                             float(line[1]),
            #                                             float(line[2]),
            #                                             float(line[3])])
            #                 if line[0].isdecimal() and line[1].replace('.','',1).isdigit() :
            #                     eyeData['Right'].append([float(line[0]),
            #                                               float(line[1]),
            #                                               float(line[2]),
            #                                               float(line[3])])
                            
            #             for m in msg_type:
            #                 if line[0] == m:
            #                       events[m].append(line[1:])
            #         else:
            #             if 'RATE' in line:
            #                 self.fs = float(line[4])
            #             if self.s_trg in line:
            #                 start = True
            #                 self.initialTimeVal = int(line[1])
            #                 if self.s_trg != 'Start_Experiment':
            #                     events['MSG'].append(line[1:])
                
        
        elif EyeTracker == "SMI":
            events = {}
            for e in self.cfg["name_epoch"]:
                events[e] = []
          
            ind_row = {"left":[],"right":[]}
         
            start = False
            for line in dat:
                if start:
                    if 'SMP' in line:
                            
                        # if float(line[ind_row["left"][2]]) == 0: # if gaze position couldn't detect, pupil position adopts instead
                        #     eyeData['Left'].append([int(line[ind_row["left"][0]]),
                        #                             float(line[ind_row["left"][1]]),
                        #                             float(line[ind_row["left"][4]]),
                        #                             float(line[ind_row["left"][5]])])
                            
                        #     eyeData['Right'].append([int(line[ind_row["right"][0]]),
                        #                             float(line[ind_row["right"][1]]),
                        #                             float(line[ind_row["right"][4]]),
                        #                             float(line[ind_row["right"][5]])])
                        #     # print("warning!")
                        # else:
                        eyeData['Left'].append([int(line[ind_row["left"][0]]),    # time stamp
                                                float(line[ind_row["left"][2]]),  # gaze x
                                                float(line[ind_row["left"][3]]),  # gaze y
                                                float(line[ind_row["left"][1]])]) # pupil
                        
                        eyeData['Right'].append([int(line[ind_row["right"][0]]),
                                                 float(line[ind_row["right"][2]]),
                                                 float(line[ind_row["right"][3]]),
                                                 float(line[ind_row["right"][1]])])
                
                            
                    for m in self.cfg["name_epoch"]:
                        if m in str(line[5]):
                             events[m].append([int(line[0]),line[5:]])
                      
                else:
                    
                    if line[0]=="Time":
                        row = []
                        for i in np.arange(len(line)-2):
                            if line[i] == "Time" or line[i] == "Type" or line[i] == "Trial":
                                row.append(line[i])
                            if line[i] == "L" or line[i] == "R":
                                row.append(line[i]+"_"+line[i+1]+"_"+line[i+2])
                      
                        for i,r in enumerate(row):
                            if r == "Time" or r == "L_Mapped_Diameter" or r == "L_POR_X" or r == "L_POR_Y":
                                ind_row["left"].append(i)
                            if r == "Time" or r == "R_Mapped_Diameter" or r == "R_POR_X" or r == "R_POR_Y" :
                                ind_row["right"].append(i)
                        
                        for i,r in enumerate(row): # pupil position
                            if r == "L_Raw_X" or r == "L_Raw_Y":
                                ind_row["left"].append(i)
                            
                            if r == "R_Raw_X" or r == "R_Raw_Y":
                                ind_row["right"].append(i)
                            
                    if 'Rate:' in line:
                        self.fs = int(line[3])
                        
                    if self.cfg['s_trg'] in line:
                        start = True
                        events[self.cfg["name_epoch"][0]].append([int(line[0]),line[5:]])
                        self.initialTimeVal = int(line[0])
                        # if s_trg != 'Start_Experiment':
                        #     events[s_trg].append(line[1:])
         
        elif EyeTracker == "VSHD":
            
            events = {}
            for e in self.cfg['s_trg']:
                events[e] = []
          
            for e in self.cfg['eventTrg']:
              events[e] = []
        
            eyeData = {"Left":[],"Right":[]}
         
            start = False
            for line in dat:
                if start:
                    if ('MovieFrame' not in line) and ("eyeA:gazeNudge" not in line) and ("eyeB:gazeNudge" not in line) and ('END' not in line):
                     
                        eyeData['Left'].append([float(line[1]),  # time stamp
                                                float(line[5]),  # gaze x
                                                float(line[6]),  # gaze y
                                                float(line[8]),  # pupil width
                                                float(line[9])]) # pupil height
                        
                        eyeData['Right'].append([float(line[1]),  # time stamp
                                                 float(line[16]),  # gaze x
                                                 float(line[17]),  # gaze y
                                                 float(line[19]),  # pupil width
                                                 float(line[20])]) # pupil height
                
                    # if len(line)==28:
                        for e in self.cfg['eventTrg']: 
                            if e in line[-1]:        
                                events[e].append(float(line[1]))
                      
                else:
                    if "FrameRate" in line:
                        self.fs = int(line[4])
                    # if len(line)==28:
                    if self.cfg['s_trg'] in line:
                        start = True
                        self.initialTimeVal = float(line[1])
                        events[self.cfg['s_trg']].append(float(line[1]))
                        # events["start"] = float(line[1])
            print(start)
        
        elif EyeTracker == "PupilCore":

            eyeData= {'Left_gaze':[],'Right_gaze':[],'Left_pupil':[],'Right_pupil':[]}
            buffer = 20
            for iEyes, lr in enumerate(["Left","Right"]):
                with open(self.cfg["rootFolder"] + 'gaze.pldata', 'rb') as f:
                    for _, payload in msgpack.Unpacker(f):
                        # if msgpack.unpackb(payload)['topic'] == "gaze.3d."+str(iEyes)+"." :
                        #     eyeData[lr+"_gaze"].append([msgpack.unpackb(payload)['timestamp'],
                        #                         msgpack.unpackb(payload)['gaze_point_3d'][0],
                        #                         msgpack.unpackb(payload)['gaze_point_3d'][1]])
                        #     eyeData[lr+"_gaze"].append([msgpack.unpackb(payload)['timestamp'],
                        #         msgpack.unpackb(payload)['norm_pos'][0],
                        #         msgpack.unpackb(payload)['norm_pos'][1]])

                        if msgpack.unpackb(payload)['topic'] == "gaze.3d.01." :
                            # eyeData[lr+"_gaze"].append([msgpack.unpackb(payload)['timestamp'],
                            #                     msgpack.unpackb(payload)['gaze_normals_3d'][str(iEyes)][0],
                            #                     msgpack.unpackb(payload)['gaze_normals_3d'][str(iEyes)][1]])
                            # eyeData[lr+"_gaze"].append([msgpack.unpackb(payload)['timestamp'],
                            #                             msgpack.unpackb(payload)['gaze_point_3d'][0],
                            #                             msgpack.unpackb(payload)['gaze_point_3d'][1]])
                            eyeData[lr+"_gaze"].append([msgpack.unpackb(payload)['timestamp'],
                                                        msgpack.unpackb(payload)['norm_pos'][0],
                                                        msgpack.unpackb(payload)['norm_pos'][1]])

                with open(self.cfg["rootFolder"] + 'pupil.pldata', 'rb') as f:
                    for _, payload in msgpack.Unpacker(f):
                        if (len(msgpack.unpackb(payload))>10) and (msgpack.unpackb(payload)['id']==iEyes):
                            eyeData[lr+"_pupil"].append([msgpack.unpackb(payload)['timestamp'],
                                                         msgpack.unpackb(payload)['diameter_3d']])
            
            with open(self.cfg["rootFolder"] + 'annotation.pldata', 'rb') as f:
                events = [[msgpack.unpackb(payload)['timestamp'],
                         msgpack.unpackb(payload)['label']]
                        for _, payload in msgpack.Unpacker(f)]
            
            self.initialTimeVal = [e[0] for e in events if e[1] == self.cfg['s_trg']][0]
            
            timeStampL = [p[0] for p in eyeData['Left_gaze']]
            timeStampR = [p[0] for p in eyeData['Right_gaze']]
        
            # self.fs = round(1 / np.diff(timeStampL)[0])
            self.fs = 120
        
            timeStampL = [round((t - self.initialTimeVal)*self.fs) for t in timeStampL]
            timeStampR = [round((t - self.initialTimeVal)*self.fs) for t in timeStampR]   
                             
            gazeXL = np.array([p[1] for p in eyeData['Left_gaze']])
            gazeXR = np.array([p[1] for p in eyeData['Right_gaze']])
        
            gazeYL = np.array([p[2] for p in eyeData['Left_gaze']])
            gazeYR = np.array([p[2] for p in eyeData['Right_gaze']])
        
            pupilDataL = np.array([p[1] for p in eyeData['Left_pupil']])
            pupilDataR = np.array([p[1] for p in eyeData['Right_pupil']])
        
            timeLen = np.max(timeStampR) if np.max(timeStampR) > np.max(timeStampL) else np.max(timeStampL)
        
            for mmName in ["pupilData","gazeX","gazeY"]:    
                if mmName == "pupilData":
                    timeStampL = [p[0] for p in eyeData['Left_pupil']]
                    timeStampR = [p[0] for p in eyeData['Right_pupil']]
                    # fs = round(1 / np.diff(timeStampL)[0])
        
                else:
                    timeStampL = [p[0] for p in eyeData['Left_gaze']]
                    timeStampR = [p[0] for p in eyeData['Right_gaze']]
                    # fs = round(1 / np.diff(timeStampL)[0])
        
                
                timeStampL = np.array([round((t - self.initialTimeVal )*self.fs) for t in timeStampL])
                if mmName == "pupilData":
                    pupilDataL = pupilDataL[timeStampL>=0]
                elif mmName == "gazeX":
                    gazeXL = gazeXL[timeStampL>=0]
                else:
                    gazeYL = gazeYL[timeStampL>=0]
                
                timeStampL = timeStampL[timeStampL>=0]
                    
                timeStampR = np.array([round((t - self.initialTimeVal )*self.fs) for t in timeStampR])
                if mmName == "pupilData":
                    pupilDataR = pupilDataR[timeStampR>=0]
                elif mmName == "gazeX":
                    gazeXR = gazeXR[timeStampR>=0]
                else:
                    gazeYR = gazeYR[timeStampR>=0]
                
                timeStampR = timeStampR[timeStampR>=0]
        
                eyeData[mmName] = np.zeros((2,timeLen+buffer))
                for iEye,lr in enumerate(['L','R']):
                    exec("eyeData[mmName][iEye,timeStamp"+ lr+"] = " + mmName + lr)
        
        elif EyeTracker == "Neon":
            
            with zipfile.ZipFile(dat) as z:
                csv_path=[f for f in z.namelist() if "3d_eye_states" in f][0]
                with z.open(csv_path) as f:
                    csv_pupil = pd.read_csv(f)

                # csv_path=[f for f in z.namelist() if "gaze.csv" == f][0]
                csv_path=[f for f in z.namelist() if "gaze.csv" in f][0]
                with z.open(csv_path) as f:
                    csv_gaze = pd.read_csv(f)

                # csv_path=[f for f in z.namelist() if "events.csv" == f][0]
                csv_path=[f for f in z.namelist() if "events.csv" in f][0]
                with z.open(csv_path) as f:
                    events = pd.read_csv(f)
            
            col_names = [col for col in csv_pupil.columns if ("eyeball" in col) or ("optical" in col)]
            
            idx = np.searchsorted(csv_gaze["timestamp [ns]"], csv_pupil["timestamp [ns]"])
            for idx2 in np.argwhere(np.diff(idx)>1).reshape(-1):

                numOfInsert=np.diff(idx)[idx2]-1
                df_insert = csv_pupil[:numOfInsert]
                df_insert[df_insert.keys()]=0
                df_insert["timestamp [ns]"]= csv_pupil.iloc[idx2]["timestamp [ns]"]+5000000*np.arange(1,numOfInsert+1)
                
                csv_pupil.iloc[idx2+1]["timestamp [ns]"]-csv_pupil.iloc[idx2]["timestamp [ns]"]
                
                csv_pupil = pd.concat(
                    [csv_pupil.iloc[:idx2+1], 
                     df_insert, 
                     csv_pupil.iloc[idx2+1:]],
                    ignore_index=True
                )
            
            # events = []
            eyeData= {'Right':{},'Left':{}}
            for iEyes in ["Left","Right"]:
                
                if f"pupil diameter {iEyes.lower()} [mm]" in csv_pupil.columns:
                    tmp_pa = csv_pupil[f"pupil diameter {iEyes.lower()} [mm]"].values
                else:
                    tmp_pa = csv_pupil["pupil diameter [mm]"].values
                    
                tmp_gx = csv_gaze["gaze x [px]"].values
                tmp_gy = csv_gaze["gaze y [px]"].values
                
                # if len(tmp_pa) > len(tmp_gx):
                #     tmp_gx = np.r_[tmp_gx,np.zeros( len(tmp_pa)-len(tmp_gx))]
                #     tmp_gy = np.r_[tmp_gy,np.zeros( len(tmp_pa)-len(tmp_gy))]
                # else:
                #     tmp_len = len(tmp_pa)
                #     tmp_pa = np.r_[tmp_pa,np.zeros( len(tmp_gx)-len(tmp_pa))]
                
                eyeData[iEyes]["timestamp"]=csv_pupil["timestamp [ns]"].values
                eyeData[iEyes]["timestamp_gaze"]=csv_gaze["timestamp [ns]"].values
                eyeData[iEyes]["gazeX"]=tmp_gx
                eyeData[iEyes]["gazeY"]=tmp_gy
                eyeData[iEyes]["pupil"]=tmp_pa
                
                for col_name in col_names:
                    if iEyes.lower() in col_name:
                        eyeData[iEyes][col_name.replace(iEyes.lower()+" ","")] = csv_pupil[col_name].values
            
            # self.initialTimeVal = csv_events["timestamp [ns]"][0]
            self.initialTimeVal = eyeData["Left"]["timestamp"][0].astype('int64')
        
            self.fs = 200
            
        elif EyeTracker == "Denso":
            
            csv_pupil = pd.read_csv(filepath_or_buffer = dat+"/gaze_front_can.csv", 
                                    encoding="ms932", sep=",",usecols=np.arange(24).tolist())

            events = []
            eyeData = {"Left":[],"Right":[]}
            for lr1, lr2 in zip(["Left","Right"],["left","right"]):
                if "eye"+lr2+"_pupildiameter" in csv_pupil.columns:
                    tmp_pa = csv_pupil["eye"+lr2+"_pupildiameter"].values
                    tmp_pa[np.isnan(tmp_pa)]=0
                    
                # else:
                #     tmp_pa = csv_pupil["pupil diameter [mm]"].values
                    
                tmp_gx = csv_pupil["gaze2d_x"].values
                tmp_gx[np.isnan(tmp_gx)]=0
                
                tmp_gy = csv_pupil["gaze2d_y"].values
                tmp_gy[np.isnan(tmp_gy)]=0
                
                if len(tmp_pa) > len(tmp_gx):
                    tmp_gx = np.r_[tmp_gx,np.zeros( len(tmp_pa)-len(tmp_gx))]
                    tmp_gy = np.r_[tmp_gy,np.zeros( len(tmp_pa)-len(tmp_gy))]
                else:
                    tmp_len = len(tmp_pa)
                    tmp_pa = np.r_[tmp_pa,np.zeros( len(tmp_gx)-len(tmp_pa))]
                
                eyeData[lr1].append([csv_pupil["timestamp"].values,
                                     tmp_gx,
                                     tmp_gy,
                                     tmp_pa])
            self.fs = 100
            self.initialTimeVal = 0
            
        elif EyeTracker == "Denso2":
            
            # csv_pupil = pd.read_csv(filepath_or_buffer = '/Users/yutasuzuki/Library/CloudStorage/Box-Box/YuutaSuzuki/Denso/N-back/results/Sub12_1_AGCなし/A/behavior_log_plus_data.csv', 
            #                         encoding="ms932", sep=",",usecols=np.arange(74).tolist())

            csv_pupil = pd.read_csv(filepath_or_buffer = dat, 
                                    encoding="ms932", sep=",",usecols=np.arange(74).tolist())

            events = []
            eyeData = {"Left":[],"Right":[]}
            for lr1, lr2 in zip(["Left","Right"],["left","right"]):
                if "eye"+lr2+"_pupildiameter" in csv_pupil.columns:
                    tmp_pa = csv_pupil["eye"+lr2+"_pupildiameter"].values
                    tmp_pa[np.isnan(tmp_pa)]=0
                    
                # else:
                #     tmp_pa = csv_pupil["pupil diameter [mm]"].values
                    
                tmp_gx = csv_pupil["gaze2d_x"].values
                tmp_gx[np.isnan(tmp_gx)]=0
                
                tmp_gy = csv_pupil["gaze2d_y"].values
                tmp_gy[np.isnan(tmp_gy)]=0
                
                if len(tmp_pa) > len(tmp_gx):
                    tmp_gx = np.r_[tmp_gx,np.zeros( len(tmp_pa)-len(tmp_gx))]
                    tmp_gy = np.r_[tmp_gy,np.zeros( len(tmp_pa)-len(tmp_gy))]
                else:
                    tmp_len = len(tmp_pa)
                    tmp_pa = np.r_[tmp_pa,np.zeros( len(tmp_gx)-len(tmp_pa))]
                
                # eyeData[lr1].append([csv_pupil["timestamp"].values,
                #                      tmp_gx,
                #                      tmp_gy,
                #                      tmp_pa,
                #                      csv_pupil["front_frame"].values])
                
                eyeData[lr1]={
                    "timestamp":csv_pupil["timestamp"].values,
                    "gazeX":tmp_gx,
                    "gazeY":tmp_gy,
                    "pupil":tmp_pa,
                    "timestamp_front_frame":csv_pupil["front_frame"].values,
                    "fileName":csv_pupil["front_sect_filename"].values
                }

                if "tobii_scene_frame" in csv_pupil.keys():
                    eyeData[lr1]["tobii_scene_frame"] = csv_pupil["tobii_scene_frame"].values
                    
                # round(1/np.mean(np.diff(csv_pupil["timestamp"].values)))
                
            self.fs = 100
            self.initialTimeVal = csv_pupil["timestamp"].values[0]
            
               # self.initialTimeVal = eyeData["Left"][0][0][0].astype('int64')
           
               # 1/(csv_pupil["timestamp"][2] - csv_pupil["timestamp"][1])
            
        #%% ------------- .asc to array ------------- %%#
        if EyeTracker != "PupilLabs":

            timeStamp = {}
            if EyeTracker == "Eyelink":
                for iEyes in ["Left","Right"]:
                    timeStamp[iEyes]=[round((t - self.initialTimeVal)*(self.fs/1000)) for t in eyeData[iEyes]["timestamp"]]
            
            elif EyeTracker == "SMI":         
                timeStampL = np.array([round(((p[0]- self.initialTimeVal)/1000) / (1 / self.fs * 1000)) for p in eyeData['Left']])
                timeStampR = np.array([round(((p[0]- self.initialTimeVal)/1000) / (1 / self.fs * 1000)) for p in eyeData['Right']])
            
            elif EyeTracker == "VSHD":
                timeStampL = np.array([p[0] for p in eyeData['Left']])
                timeStampR = np.array([p[0] for p in eyeData['Right']])
                
                timeStampL = [round(round((t - self.initialTimeVal) * self.fs)) for t in timeStampL]
                timeStampR = [round(round((t - self.initialTimeVal) * self.fs)) for t in timeStampR]
             
              
            elif EyeTracker == "Neon":
                for iEyes in ["Left","Right"]:
                    timeStamp[iEyes]=((np.array(eyeData[iEyes]["timestamp"]) - self.initialTimeVal)*(10**(-9))*self.fs).astype(int)
                    timeStamp[iEyes+"_gaze"]=((np.array(eyeData[iEyes]["timestamp_gaze"]) - self.initialTimeVal)*(10**(-9))*self.fs).astype(int)
                    # timeStamp[iEyes]=((np.array(eyeData[iEyes]["timestamp"]) - initialTimeVal)*(10**(-9))*fs).astype(int)
                    # timeStamp[iEyes+"_gaze"]=((np.array(eyeData[iEyes]["timestamp_gaze"]) - initialTimeVal)*(10**(-9))*fs).astype(int)

            elif "Denso" in EyeTracker:
                for iEyes in ["Left","Right"]:
                    # timeStamp[iEyes]=[((t-initialTimeVal)*100) for t in eyeData[iEyes]["timestamp"]]
                    timeStamp[iEyes]=[round((t-self.initialTimeVal)*100) for t in eyeData[iEyes]["timestamp"]]
                    timeStamp[iEyes]=np.arange(len(eyeData[iEyes]["timestamp"]))
                
                # a=np.diff(eyeData[iEyes]["timestamp"])*10000
                
            timeLen = np.max(timeStamp["Right"]) if np.max(timeStamp["Right"]) > np.max(timeStamp["Left"]) else np.max(timeStamp["Left"])
            
            if EyeTracker == "Neon":
                for mmName in ['gazeX','gazeY','pupil',
                               'eyeball center x [mm]','eyeball center y [mm]','eyeball center z [mm]',
                               'optical axis x','optical axis y','optical axis z']:
                    # eyeData[mmName] = np.zeros((2,timeLen+1))
                    timeLen = len(eyeData["Right"][mmName])
                    # timeLen = len(eyeData["Right"]["gazeX"])
                    eyeData[mmName] = np.zeros((2,timeLen))
                    
                    for iEye,eye in enumerate(["Left","Right"]):
                        # if "gaze" in mmName:
                        #     # print(eyeData[eye][mmName].shape)
                        #     print(timeStamp[eye+"_gaze"])
                        #     eyeData[mmName][iEye,timeStamp[eye+"_gaze"]]=eyeData[eye][mmName]
                            
                        # else:
                        #     eyeData[mmName][iEye,timeStamp[eye]]=eyeData[eye][mmName]
                        if "gaze" in mmName:
                            eyeData[mmName][iEye,range(timeLen)]=eyeData[eye][mmName]
                        else:
                            eyeData[mmName][iEye,range(timeLen)]=eyeData[eye][mmName]
                            
            else:
                for mmName in ['gazeX', 'gazeY', 'pupil']:
                    eyeData[mmName] = np.zeros((2,timeLen+1))
                    for iEye,eye in enumerate(["Left","Right"]):
                        eyeData[mmName][iEye,timeStamp[eye]]=eyeData[eye][mmName]
                
        if "Denso" in EyeTracker:
            eyeData["timeStamp"] = timeStamp
            eyeData["fileName"] = [[] for _ in range(np.max(timeStamp["Right"])+1)]
            for i,t in enumerate(timeStamp["Right"]):
                eyeData["fileName"][t] = eyeData["Left"]["fileName"][i]
            
        interpolatedArray = []
        interpolatedArray.append(np.argwhere(eyeData["pupil"][0,]==0).reshape(-1))
        interpolatedArray.append(np.argwhere(eyeData["pupil"][1,]==0).reshape(-1))
            
        #%% ------------- resampling -------------
        figCount = 1
        if self.cfg["visualization"]:
            fig, ax = plt.subplots(self.figSize[0],self.figSize[1],figsize=(20,20))
            # fig.suptitle("Overview of pre-processing")
            
        self.dataLen = eyeData["pupil"].shape[1]
        pupil_withoutInterp = eyeData["pupil"].copy()
        
        eyeData["pupilData_original"] = eyeData["pupil"].copy()
        
        # if self.fs > 250: # temporary down-sampled to 250Hz for blink interpolation
        #     for mmName in ['gazeX', 'gazeY', 'pupil']:
        #         eyeData[f"{mmName}_downsampled"] = re_sampling(eyeData[mmName].copy(),round(self.dataLen*(250/self.fs)))
            
        #     eyeData["pupil"] = eyeData["pupil_downsampled"]
        
        # else:
        #     for mmName in ['gazeX', 'gazeY', 'pupil']:
        #         eyeData[f"{mmName}_downsampled"] = eyeData[mmName]

        #%% ------------- outlier (> 3*sigma) are replaced by zero -------------
        
        eyeData["numOfZero_original"] = len(np.argwhere(eyeData["pupil"][0,]==0))  / eyeData["pupil"].shape[1]
    
        if EyeTracker == "VSHD":
            for iEyes in np.arange(eyeData["pupil"].shape[0]):
             
                ind = np.argwhere(abs(eyeData["pupil"][iEyes,:]) > 1).reshape(-1)
            
                for mmName in ["pupilData","gazeX","gazeY"]:
                    eyeData[mmName][iEyes,ind] = 0  
        
        if EyeTracker == "Neon":   
            eyeData["pupil"] = zeroInterp_oneshot(eyeData["pupil"].copy())

        # for irow,mmName in enumerate(["pupil","gazeX","gazeY","gazeX_downsampled","gazeY_downsampled"]):    
        for irow,mmName in enumerate(["pupil","gazeX","gazeY"]):
        # for irow,mmName in enumerate(["gazeX","gazeY"]):
            for iEyes in np.arange(eyeData["pupil"].shape[0]):

                if EyeTracker == "Neon":
                    tmp=eyeData[mmName][iEyes,:]
                    d = np.diff(tmp[tmp!=0])
                
                else:
                    d = np.diff(eyeData[mmName][iEyes,:])
               
                # d = np.diff(eyeData[mmName])
                # q75, q25 = np.percentile(d[d!=0], [75 ,25])
                q75, q25 = np.percentile(d, [75 ,25])
                IQR = q75 - q25
    
                lower = q25 - IQR*1.5
                upper = q75 + IQR*1.5
                # lower = q25 - IQR*10
                # upper = q75 + IQR*10
                
                # sigma = np.std(d)
               
                # d_witout_outliar = d[(d!=0) & (d > -sigma) & (d < sigma)]
                d_witout_outliar = d[(d > lower) & (d < upper)]
                # d_witout_outliar = d
            
                param = norm.fit(d_witout_outliar)
                
                if EyeTracker == "Neon":
                    (a,b) = norm.interval(0.9, loc=param[0], scale=param[1])
                    # (a,b) = norm.interval(0.99, loc=param[0], scale=param[1])
                #     a=-1
                #     b=1
                else:
                    # (a,b) = norm.interval(0.999, loc=param[0], scale=param[1])
                    # (a,b) = norm.interval(0.9, loc=param[0], scale=param[1])
                    (a,b) = norm.interval(0.99, loc=param[0], scale=param[1])

                    # (a,b) = norm.interval(0.999, loc=param[0], scale=param[1])
         
                if mmName == "pupil":
                    pa=a
                    pb=b
                # (a,b) = norm.interval(0.95, loc=param[0], scale=param[1])
                # (a,b) = norm.interval(0.8, loc=param[0], scale=param[1])
                
                ind = np.argwhere(abs(np.diff(eyeData[mmName][iEyes,:])) < b).reshape(-1)
                # print(a)
                # print(b)
                arr = eyeData[mmName][iEyes, ind]
                if arr.size == 0:
                    print(f'Average without interp. {iEyes} {mmName} = NaN (no samples)')
                else:
                    print(f'Average without interp. {iEyes} {mmName} = {np.round(arr.mean(), 2)}')

                ind = np.argwhere(abs(np.diff(eyeData[mmName][iEyes,:])) > b).reshape(-1)
                # print(len(ind))
                eyeData[mmName][iEyes,ind] = 0  
            
            # plt.figure()
            # plt.plot(eyeData["pupil"][0,:])
            # plt.ylim([0,1])
            # plt.figure()
            # plt.plot(eyeData["pupil"][1,:])
            # plt.ylim([0,1])
                
            if self.cfg["visualization"]:
                
                ######## histgram ########
                # plt.hist(d, bins=1000, alpha=1, color='g',density=True, histtype="stepfilled")
                # plt.hist(d_witout_outliar, alpha=0.5,density=True, histtype="stepfilled")
                
                ax[irow, 0].hist(d_witout_outliar, alpha=0.5,density=True, histtype="stepfilled",bins=100)
                xmin, xmax = ax[irow, 0].get_xlim()

                x = np.linspace(xmin, xmax, 100)
                
                p = norm.pdf(x, loc=param[0], scale=param[1])
                ax[irow, 0].plot(x, p, 'k', linewidth=2)
                title = "Fit std = %.2f" % (param[1])
                # plt.title(title)
                ax[irow, 0].axvline(x=a, ls='--', color='#2ca02c')
                ax[irow, 0].axvline(x=b, ls='--', color='#2ca02c')

                ax[irow, 0].set_xlabel("Velocity")
                ax[irow, 0].set_ylabel("Prob.")
                ax[irow, 0].set_title("Threshold")
                # plt.ylim([0,100])
                # ax[irow, 0].set_xlim([pa*3,pb*3])

                # figCount += 1

                ######## velocity ########
                # plt.subplot(self.figSize[0],self.figSize[1],figCount)
                # print(len(d))
                # plt.plot(d.T,'.',alpha=0.5, markersize=1)
                ax[irow, 1].plot(d.T,'.',alpha=0.5, markersize=1)
                ax[irow, 1].axhline(y=a, ls='--', color='#2ca02c')
                ax[irow, 1].axhline(y=b, ls='--', color='#2ca02c')  
                # # plt.xlim([110000,111000])
                ax[irow, 1].set_xlabel("Time")
                ax[irow, 1].set_ylabel("Velocity")
                ax[irow, 1].set_title("Changes in pupil velocity")
                # ax[irow, 1].set_ylim([pa*3,pb*3])

                # figCount += 1
                
                # ######## time-course ########
                # plt.subplot(self.figSize[0],self.figSize[1],figCount)
                
                tmp = eyeData[mmName][0,].copy()
                tmp[ind]= np.nan
                tmp[tmp==0] = np.nan
                ax[irow, 2].plot(tmp,'.', label="Left", markersize=1)
                
                tmp = eyeData[mmName][1,].copy()
                tmp[ind]= np.nan
                tmp[tmp==0] = np.nan
                ax[irow, 2].plot(tmp,'.', label="Right", markersize=1)
                # plt.legend()
                # # plt.xlim([0,10*self.fs])
                # # plt.ylim([0,1])
                ax[irow, 2].set_xlabel("Time")
                ax[irow, 2].set_ylabel(mmName)
                ax[irow, 2].set_title("Changes in '"+mmName+"'")

                # # plt.xlim([110000,111000])
                
                # figCount += 1
                
                # ######## time-course ########
                # plt.subplot(self.figSize[0],self.figSize[1],figCount)
                
                tmp = eyeData[mmName][0,].copy()
                tmp[ind]= np.nan
                tmp[tmp==0] = np.nan
                ax[irow, 3].plot(tmp,'.', label="Left", markersize=1)
                
                tmp = eyeData[mmName][1,].copy()
                tmp[ind]= np.nan
                tmp[tmp==0] = np.nan
                ax[irow, 3].plot(tmp,'.', label="Right", markersize=1)
                
                # plt.legend()
                ax[irow, 3].set_xlim([10*self.fs,20*self.fs])
                ax[irow, 3].set_title("Changes in '"+mmName+"' from 10-20sec")
                ax[irow, 3].legend()
                # # plt.xlim([110000,111000])
                
                # # plt.ylim([0,1])
                # plt.xlabel("Time")
                # plt.ylabel(mmName)
                
                # figCount += 1
        eyeData["numOfZero_th"] = len(np.argwhere(eyeData["pupil"][0,]==0)) / eyeData["pupil"].shape[1]

        # return eyeData

        if EyeTracker == "Neon":   
            eyeData["pupil"] = zeroInterp_oneshot(eyeData["pupil"].copy())
            
        if self.cfg["visualization"]:
            # plt.subplot(self.figSize[0],self.figSize[1],figCount)
            # plt.plot(eyeData["gazeX"].T,eyeData["gazeY"].T,'.')
            # plt.xlabel("gaze x")
            # plt.ylabel("gaze y")
            
            # plt.fig.set_figheight(50)
            # plt.fig.set_figwidth(6)

            # figCount += 1
            
            if not os.path.exists(os.path.dirname(self.cfg["fName"])):
                os.makedirs(os.path.dirname(self.cfg["fName"]),exist_ok=True)
            
            plt.savefig(f"{self.cfg['fName']}_preprocessing.png", 
                        bbox_inches="tight",
                        pad_inches=0.2, 
                        dpi=300)
         
        # eyeData["pupil"] = pupilData
        # eyeData["xData"] = xData
        # eyeData["yData"] = yData
        eyeData["interpolatedArray"] = interpolatedArray
      
        return eyeData,events,self.initialTimeVal,self.fs
    
    def getAve(self, pupilData, eyeTracker="Eyelink"):
        
        tmp_p = abs(pupilData.copy())
        
        if self.cfg["mmFlag"]:
            if eyeTracker == "VSHD":
                tmp_p = tmp_p * getmm(eyeTracker)[0]
                
            else:
                tmp_p = (tmp_p/256)**2*np.pi
                tmp_p = np.sqrt(tmp_p) * au2mm(700) 
            
        ind_nonzero = np.argwhere(tmp_p[0,:] != 0).reshape(-1)
        ave_left = np.mean(tmp_p[0,ind_nonzero])
        sigma_left = np.std(tmp_p[0,ind_nonzero])   
        
        ind_nonzero = np.argwhere(tmp_p[1,:] != 0).reshape(-1)
        ave_right = np.mean(tmp_p[1,ind_nonzero])
        sigma_right = np.std(tmp_p[1,ind_nonzero])       
        
        ave = np.mean([ave_left,ave_right])
        sigma = np.mean([sigma_left,sigma_right])
        
        print('Normalized by mu = ' + str(np.round(ave,4)) + ' sigma = ' + str(np.round(sigma,4)))
     
        return ave,sigma
    
    def regressout(self,df,descrip):

        model = smf.ols(descrip, data=df)
        result = model.fit()
        return result.resid + result.params["Intercept"]
        
    def blinkInterp(self,eyeData, eyeTracker="Eyelink"):
        
        pupilData_bef = eyeData["pupilData_original"]
        # pupilData = eyeData["pupilData_original"]
        pupilData = eyeData["pupil"]

        # %%

        # pupilData = eyeData["pupilData_original"]
        
        pupilData_lowpass = pupilData.copy()
        iEyes=0

        def moving_avg_centered(x, window):
            kernel = np.ones(window) / window
            return np.convolve(x, kernel, mode='same')

        for iEyes in np.arange(2):
            pupilData_lowpass[iEyes,pupilData_lowpass[iEyes,:]!=0] = lowpass_filter(pupilData[iEyes,pupilData[iEyes,:]!=0], 40, self.fs)
            # pupilData_lowpass[iEyes,pupilData_lowpass[iEyes,:]!=0] = lowpass_filter(pupilData[iEyes,pupilData[iEyes,:]!=0], 40, 200)
            # pupilData_lowpass[iEyes,] = moving_avg_centered(pupilData_lowpass[iEyes,],10)
        
        pupilData_lowpass[pupilData_lowpass<0]=0
        
        # pupilData=pupilData_lowpass
        # tmp_pupilData = eyeData["pupil"]
        # pupilData_intep = zeroInterp(pupilData.copy(),self.fs,self.fs*0.04)
        pupilData_intep = zeroInterp(pupilData_lowpass.copy(),self.fs,self.fs*0.04)
        # pupilData_intep = zeroInterp(pupilData_lowpass.copy(),200,200*0.04)
        # pupilData_intep = zeroInterp(pupilData.copy(),fs,fs*0.04)

        # plt.figure()
        # plt.plot(np.arange(pupilData.shape[1])*(1/fs),pupilData_intep["pupilData"][0,:].T,alpha=0.8)
        # plt.plot(np.arange(pupilData.shape[1])*(1/fs),pupilData[0,:].T,alpha=0.8,label="raw")
        # plt.plot(np.arange(pupilData.shape[1])*(1/fs),pupilData_lowpass[0,:].T,alpha=0.8,label="lowpass")
        # # plt.plot(pupilData.T,alpha=0.5)
        # plt.xlim(200,240)
        # plt.legend()

        # %%

        # plt.xlim(180,183)
        # plt.xlim(190,192)
        # plt.ylim(30,70)
        # plt.ylim(0,10)

        gazeX = eyeData["gazeX"]
        gazeY = eyeData["gazeY"]

        # if self.fs > 250: # temporary down-sampled to 250Hz for blink interpolation
        #     # pupilData = zeroInterp(pupilData.copy(),250,5)
        #     pupilData_intep = zeroInterp(pupilData.copy(),250,5)
        #     # pupilData_intep = zeroInterp2(pupilData.copy(),250)
            
        # else:
        #     pupilData_intep = zeroInterp(pupilData.copy(),self.fs,(self.fs*0.04))
        #     # pupilData_intep = zeroInterp(pupilData.copy(),250,10)

        print(f"Interpolated array = {pupilData_intep['interpolatedArray']} out of {pupilData_intep['pupilData'].shape[1]}")
        
        # if interplated array is 60% out of whole data
        interpArray = np.array(pupilData_intep['interpolatedArray']) / pupilData_intep['pupilData'].shape[1]
        
        if (min(np.array(pupilData_intep['interpolatedArray']))/pupilData_intep['pupilData'].shape[1]) > 0.4:
            rejectFlag = True
        else:
            rejectFlag = False
            
        if pupilData_intep['interpolatedArray'][0] < pupilData_intep['interpolatedArray'][1]:
        # if pupilData_intep['zeroArray'][0] < pupilData_intep['zeroArray'][1]:
            betterEye = 'L'
        else:
            betterEye = 'R'
            
        zeroArray = np.zeros((2,pupilData.shape[1]))
        for iEyes in np.arange(2):
            zeroArray[iEyes,pupilData_intep["zeroArray"][iEyes]] = 1
            
        if self.cfg['usedEye'] == 1:#### better eye
            if pupilData_intep['interpolatedArray'][0] < pupilData_intep['interpolatedArray'][1]:
            # if pupilData_intep['zeroArray'][0] < pupilData_intep['zeroArray'][1]:
                pupilData = pupilData_intep['pupilData'][0,:].reshape(1,pupilData_intep['pupilData'].shape[1])
                gazeX = gazeX[0,:].reshape(1,gazeX.shape[1])
                gazeY = gazeY[0,:].reshape(1,gazeY.shape[1])
                usedEye = 'L'
                mmName = ['Left']
                 
            else:
                pupilData = pupilData_intep['pupilData'][1,:].reshape(1,pupilData_intep['pupilData'].shape[1])
                gazeX = gazeX[1,:].reshape(1,gazeX.shape[1])
                gazeY = gazeY[1,:].reshape(1,gazeY.shape[1])
                usedEye = 'R'
                mmName = ['Right']
                
        elif self.cfg['usedEye'] == 'L': 
            pupilData = pupilData_intep['pupilData'][0,:].reshape(1,pupilData_intep['pupilData'].shape[1])
            gazeX = gazeX[0,:].reshape(1,gazeX.shape[1])
            gazeY = gazeY[0,:].reshape(1,gazeY.shape[1])
            usedEye = 'L'
            mmName = ['Left']
            
        elif self.cfg['usedEye'] == 'R':
            pupilData = pupilData_intep['pupilData'][1,:].reshape(1,pupilData_intep['pupilData'].shape[1])
            gazeX = gazeX[1,:].reshape(1,gazeX.shape[1])
            gazeY = gazeY[1,:].reshape(1,gazeY.shape[1])
            usedEye = 'R'
            mmName = ['Right']
       
        else: # both eyes
            pupilData = pupilData_intep['pupilData']
            usedEye = 'both'
            mmName = ['Left','Right']
   
        if self.cfg["usedEye"] == 0: #### Both
            for iEyes in np.arange(pupilData.shape[0]): 
                zeroInd = np.argwhere(gazeX[iEyes,:] == 0).reshape(-1)
                gazeX[1-iEyes,zeroInd] = 0
               
                zeroInd = np.argwhere(gazeY[iEyes,:] == 0).reshape(-1)
                gazeY[1-iEyes,zeroInd] = 0
                
                zeroInd = np.argwhere(pupilData_intep[iEyes,:] == 0).reshape(-1)
                pupilData_intep[1-iEyes,zeroInd] = 0
        
        print(f"{mmName} was used.")
        
        if self.cfg["visualization"]:
            
            # fig, ax = plt.subplots(1, 1, figsize=(20, 10), sharex=True)
            plt.figure()
            iEyes = 0 if usedEye=="L" else 1
            # for iEyes in range(pupilData_bef.shape[0]):
            x = np.arange(pupilData_bef.shape[1]) / self.fs
            # print(pupilData_bef.shape)
            # print(pupilData["pupilData"].shape)
            df = pd.DataFrame({
                "Time[s]": np.r_[x, x],
                "Pupil[mm]": np.r_[pupilData_bef[iEyes, :], pupilData[0, :]],
                "Label": (["Before interpolation"] * len(x)) + (["After interpolation"] * len(x))
            })
        
            sns.lineplot(data=df, x="Time[s]", y="Pupil[mm]", hue="Label",
                         alpha=0.8, linewidth=0.5)
        
            plt.ylabel("Pupil [mm]")
            plt.grid(True, alpha=0.2)
            
            plt.xlabel("Time [s]")
            plt.tight_layout()
            
            plt.savefig(f"{self.cfg['fName']}_interp.png", 
                        bbox_inches="tight",
                        pad_inches=0.2, 
                        dpi=300)
            
        # gazeX = re_sampling(gazeX.copy(),self.dataLen)
        # gazeY = re_sampling(gazeY.copy(),self.dataLen)
        pupilData = re_sampling(pupilData.copy(),self.dataLen)
        # zeroArray = re_sampling(zeroArray.copy(),self.dataLen)
        # zeroArray = np.repeat(zeroArray.copy(), self.fs/250)
        zeroArray = np.repeat(zeroArray.copy(), 2)
        if len(zeroArray) > pupilData.shape[1]:
            zeroArray = zeroArray[:pupilData.shape[1]]
        else:
            zeroArray = np.r_[zeroArray,np.zeros(pupilData.shape[1]-len(zeroArray))]
        
            
        #%% -------------  transform to mm ------------- %%#
        if self.cfg["mmFlag"]:
            if eyeTracker == "VSHD":
                pupilData = pupilData * getmm(eyeTracker)[0]
            else:
                pupilData = abs(pupilData)
                
                if eyeTracker=="Eyelink":
                    if self.Method=="DIAMETER":
                        if usedEye == 'L':
                            pupilData[0,:] = pupilData[0,:] * au2mm()[0]
        
                        elif usedEye == 'R':
                            pupilData[0,:] = pupilData[0,:] * au2mm()[1]
                            
                        else:
                            for iEye in np.arange(2):
                                pupilData[iEye,:] = pupilData[iEye,:] * au2mm()[iEye]
                                
                    else:
                        pupilData = (pupilData/256)**2*np.pi
                        pupilData = np.sqrt(pupilData) * au2mm(700)
   
    
        #%% -------------  filtering ------------- %%# 
        if len(self.cfg['WID_FILTER']) > 0:
            pupilData = lowpass_filter(pupilData, self.cfg['WID_FILTER'][1], self.cfg['SAMPLING_RATE'])
            
            
        #%% -------------  data save ------------- %%#
        
        eyeData = {'pupilData':pupilData,
                   'gazeX':gazeX,
                   'gazeY':gazeY,
                   "zeroArray":zeroArray,
                   "numOfZero_original":eyeData["numOfZero_original"],
                   "numOfZero_th":eyeData["numOfZero_th"],
                   # 'MS':ev,
                   # 'MS_events':ms,
                   'usedEye':usedEye,
                   'betterEye':betterEye,
                   'rejectFlag':rejectFlag,
                   # 'data_test':data_test,
                   # 'data_control_bef':data_control_bef,
                   # 'data_control_aft':data_control_aft,
                   'interpRate':interpArray
                }
            
        return eyeData
#     pupilData = re_sampling(pupilData.copy(),dataLen)

    def pupilNorm(self,pupilData,ave,sigma):
        
        return (pupilData - ave) / sigma
        
    def showResults(self,events,eyeData):
        
        pupilData = eyeData["pupil"]
        #%% -------------  show results ------------- %%#
     
        # st = [[int(int(e[0])- self.initialTimeVal),e[1]] for e in events['MSG'] if 'Start' in e[1]]       
        # ed = [[int(int(e[0])- self.initialTimeVal),e[1]] for e in events['MSG'] if 'End' in e[1]]       
        
        # if (len(st) == 0) | (len(ed) == 0):
        #     st.append([0])
        #     ed.append([pupilData.shape[1]])
        
        # for iEyes in np.arange(pupilData.shape[0]): 
        #     print('Average ' + self.mmName[iEyes] + ' pupil size = ' + str(np.round(pupilData_intep[iEyes,st[0][0]:ed[0][0]].mean(),2)))
            
        if self.cfg["visualization"]:
            # plt.subplot(figSize[0],figSize[1],figCount)
            # figCount += 1
            plt.figure()
            plt.plot(pupilData.T)
            # plt.plot(pupil_withoutInterp[iEyes,st[0][0]:ed[0][0]].T,'k',alpha=0.2)
            # plt.ylim([5000,9000])
             
            # plt.subplot(figSize[0],figSize[1],figCount)
            # figCount += 1
            # plt.plot(pupilData_intep[iEyes,st[0][0]:ed[0][0]].T)
            # plt.plot(pupil_withoutInterp[iEyes,st[0][0]:ed[0][0]].T,'k',alpha=0.2)
            # plt.ylim([5000,9000])
            # plt.xlim([45000,50000])
            
            # plt.subplot(figSize[0],figSize[1],figCount)
            # figCount += 1
            # plt.plot(np.diff(pupilData_intep[iEyes,st[0][0]:ed[0][0]]).T)
            
            # plt.hlines(upsilon, 0, len(xData), "red", linestyles='dashed')
            # plt.savefig("./img.pdf")
        # print('upsilon = ' + str(np.round(upsilon,4)) + ', std = ' + str(np.round(np.nanstd(v),4)))

    #%% -------------  data plot ------------- %%#
    
    # pupilData = np.mean(pupilData,axis=0)
    # xData = np.mean(xData,axis=0)
    # yData = np.mean(yData,axis=0)
    
    
    # plt.plot(pupilData.T,color="k")
    # plt.ylim([0,10000])
    
    # plt.subplot(2,3,2)
    # plt.plot(np.diff(pupilData).T,color="k")
    # plt.ylim([-50,50])
    
    # plt.subplot(1,3,2)
    # plt.plot(pupilData.T)
    # plt.xlim([200000, 210000])
    # # plt.ylim([20000,10000])
    
    # plt.subplot(1,3,3)
    # plt.plot(np.diff(pupilData).T)
    # plt.xlim([200000, 210000])
    # plt.ylim([-50,50])
    # plt.subplot(2,3,4)
    # plt.plot(pupilData.T,color="k")
    # plt.xlim([500000, 550000])
    # plt.ylim([0,10000])
    
    # plt.subplot(2,3,5)
    # plt.plot(pupilData.T,color="k")
    # plt.xlim([1000000, 1050000])
    # plt.ylim([0,10000])
    
    # plt.subplot(2,3,6)
    # plt.plot(pupilData.T,color="k")
    # plt.xlim([2000000, 2050000])
    # plt.ylim([0,10000])

    # if normFlag:
    #     pupilData = (pupilData - ave) / sigma
              
    # if len(filt) > 0:
    #     pupilData = butter_bandpass_filter(pupilData, filt[0], filt[1], fs, order=4)

    # return eyeData,events,self.initialTimeVal,int(fs)
