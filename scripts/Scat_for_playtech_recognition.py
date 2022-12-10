# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3.8.5 ('base')
#     language: python
#     name: python3
# ---

import time
import os
from tqdm import tqdm
import scipy.io, soundfile
from fnmatch import fnmatch
#import librosa
import pandas as pd
import numpy as np
import re, collections, requests
import matplotlib.pyplot as plt
import subprocess
import warnings
warnings.filterwarnings('ignore')

# # Download CBFdataset

# +
# ACCESS_TOKEN = "replace this with your access token"
# record_id = "replace this with your record"

# r = requests.get(f"https://zenodo.org/api/records/5744336", params={'access_token': ACCESS_TOKEN})
# download_urls = [f['links']['self'] for f in r.json()['files']]
# filenames = [f['key'] for f in r.json()['files']]

# for filename, url in zip(filenames, download_urls):
# #     print("Downloading:", filename)
#     r = requests.get(url, params={'access_token': ACCESS_TOKEN})
#     with open(filename, 'wb') as f:
#         f.write(r.content)

# +
# # unzip dataset files
# from zipfile import ZipFile
# # !mkdir CBFdataset
# with ZipFile('CBFdataset.zip', 'r') as zipObj:  
#     zipObj.extractall('CBFdataset/') 
# # !rm CBFdataset.zip

# +
# check wav files of the dataset
base_dir = 'CBFdataset_PETS/'
target = "*.wav"
wav_files = []  
for path, subdirs, files in os.walk(base_dir):
    for name in files:
        if fnmatch(name, target):
            wav_files.append(os.path.join(path, name))  
print('Number of audio files:', format(len(wav_files)))

# save file names for convenient feature extraction by matlab
# with open('file_names.txt', 'w') as f:
#     for item in wav_files:
#         f.write("%s\n" % item)
        
# check duration of the dataset
total_len = 0
for k in range(len(wav_files)):
    x, sr = soundfile.read( wav_files[k])
    total_len = total_len + x.shape[0]/sr

print("Total duration of the dataset: %.2f h." % (total_len/3600))

# +
with open('file_names.txt', 'r') as f:
    wav_files = f.readlines()
    wav_files = [w.strip() for w in wav_files]
# -

# # Feature extraction

# ## F0 extraction

# +
# frame_length = 2048; hop_length = 128
# f0 = {'file'+str(k):[] for k in range(len(wav_files))}

# t0 = time.time()
# for k in tqdm(range(len(wav_files))):
#     y, sr = soundfile.read(wav_files[k])
#     y = np.mean(y, 1)
#     f0_file, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C8'), sr=sr,
#                                                      frame_length = frame_length, hop_length = hop_length)
#     f0['file'+str(k)] = f0_file

# from scipy.io import savemat
# savemat("F0_trajectory_default_C2C8_hop128.mat", f0)
# print('F0 extraction time:%.2f hours.' % ((time.time() - t0)/3600))
# -

# ## AdaTS+AdaTRS feature extraction

# +
# # extract AdaTS+AdaTRS feature in matlab
# t0 = time.time()
# subprocess.call(["matlab",
#                  "-r",
#                  "AdaTS_AdaTRS_PMT_extraction",
#                  "-nodisplay",
#                  "-nodesktop"])

# print('Feature extraction time:%.2f hours.' % ((time.time() - t0)/3600))

# +
# # load extracted feature
# adapt_time = scipy.io.loadmat('AdaTS_AdaTRS_PMT_feature.mat')['fileFeatures_time'][0,:]
# adapt_timerate = scipy.io.loadmat('AdaTS_AdaTRS_PMT_feature.mat')['fileFeatures_timerate'][0,:]

# print(adapt_time.shape, adapt_timerate.shape)

# adapt = []
# for k in range(adapt_time.shape[0]):
#     adapt.append(np.vstack((adapt_time[k], adapt_timerate[k])))
    
# adapt_time_dim = adapt_time[0].shape[0]
# del(adapt_time,adapt_timerate)
# print(adapt_time_dim, adapt[0].shape[0])
# -

# ## dJTFS feature extraction

# +
# # extract dJTFS-avg feature in matlab
# t0 = time.time()
# subprocess.call(["matlab",
#                  "-r",
#                  "dJTFS_avg_PET_extraction",
#                  "-nodisplay",
#                  "-nodesktop"])

# +
# print('Feature extraction time:%.2f hours.' % ((time.time() - t0)/3600))
# -

# load extracted feature
joint = scipy.io.loadmat('matlab/dJTFS_acciacatura.mat')['fileFeatures'][0, :]
print("Num files:", len(joint))
print("Feature dims (of 3rd file):", joint[2].shape)

# +
# plt.figure(dpi=150)
# plt.imshow(joint[0], aspect="auto")
# plt.title("10G_Iso_Glissando")
# #plt.savefig("10G_Iso_Glissando.png")
# plt.show()
# #np.save("10G_Iso_Glissando.npy", joint[0])

# +
# plt.figure(dpi=150)
# plt.imshow(joint[1], aspect="auto")
# plt.title("10G_Iso_Portamento")
# #plt.savefig("10G_Iso_Portamento.png")
# plt.show()
# #np.save("10G_Iso_Portamento.npy", joint[1])

# +
# plt.figure(dpi=150)
# plt.imshow(joint[2], aspect="auto")
# plt.title("10G_Iso_Acciacatura")
# #plt.savefig("10G_Iso_Acciacatura.png")
# plt.show()
# #np.save("10G_Iso_Acciacatura.npy", joint[2])
# -

# ## feature concatenation

# +
# joint context
context = 2

joint_contexted = [None] * len(joint)
joint_contexted = np.array(joint_contexted)
for k in range(len(joint)):
#     duplicate adapt feature to have the same number of frames
    #adapt[k] = np.repeat(adapt[k], 2, axis=1)
    #adapt[k] = adapt[k][:,:joint[k].shape[1]]
    joint_contexted[k] = np.vstack((joint[k], joint[k]))
    for m in range(context,joint[k].shape[1]-context):  # mean and std of 5 frames to take account context information for PETs
        joint_contexted[k][0:joint[k].shape[0],m] = np.mean(joint[k][:,m-context:m+context+1], axis=1)
        joint_contexted[k][joint[k].shape[0]:, m] = np.std(joint[k][:,m-context:m+context+1], axis=1)
    #joint_contexted[k] = np.vstack((adapt[k], joint_contexted[k]))

feature = joint_contexted
#del(adapt, joint, joint_contexted)
#print(feature.shape, feature[21].shape)
print("Mean + stdev of features, dim:", feature.shape)
# -

print(feature[0].shape, feature[1].shape)
print(sum([f.shape[1] for f in feature]))

# +
# plt.figure(dpi=150)
# plt.imshow(feature[2], aspect="auto")
# -

# ## load annotations

# +
# prepare annotation from .csvs
# tech_name = np.array(['Tremolo', 'Acciacatura', 'Glissando', 'Trill', 'FT', 'Vibrato', 'Portamento'])
tech_name = np.array(['Acciacatura', 'Glissando', 'Portamento'])
anno_files = [None]

for k in range(len(wav_files)):
    if wav_files[k].split('/')[2] == 'Iso':
        anno_files.append(wav_files[k].replace('.wav', '.csv'))
    elif wav_files[k].split('/')[2] == 'Piece': # 'Piece'
        for m in range(len(tech_name)):
            if os.path.exists(wav_files[k][:-4] + "_tech_" + tech_name[m] + '.csv'):
                anno_files.append(wav_files[k][:-4] + "_tech_" + tech_name[m] + '.csv')
                
anno_files = anno_files[1:]
print(len(anno_files))

# +
feature_conca = np.zeros((feature[0].shape[0],1))
file_id = 0
player_id = 0

for k in range(len(feature)):
    # connect all features
    feature_conca = np.hstack((feature_conca, feature[k]))
    # gte file ID
    file_id = np.hstack((file_id, np.ones((feature[k].shape[1]), dtype=int) * k))
    # get player ID
    if wav_files[k].split('/')[1][-1] == '0':
        player_id = np.hstack((player_id, np.ones((feature[k].shape[1]), dtype=int) * 10)) 
    else:
        player_id = np.hstack((player_id, np.ones((feature[k].shape[1]), dtype=int) * int(wav_files[k].split('/')[1][-1]))) 
        
player_id = player_id[1:]
file_id = file_id[1:]
print(feature_conca.shape)
feature_conca = np.transpose(feature_conca)
print(feature_conca.shape)
feature_conca = feature_conca[1:]
print(feature_conca.shape)
# -

# scattering params
sr = 44100
T = 2**14  # PMT T=15, PET T=14 => PMT duplicated
oversampling = 2
hop_sample = T/(2**oversampling)
print('frame size: %sms' % (int(hop_sample/44100*1000)))

# +
import re
label_id = {k:[None] for k in range(len(feature))}

# get label ID
for k in range(len(feature)):
    label_id[k] = np.zeros((len(tech_name), feature[k].shape[1]),dtype=int)
    if wav_files[k].split('/')[2] == 'Iso':
        anno_files = wav_files[k].replace('.wav', '.csv')
        file_anno = pd.read_csv(anno_files)
        file_onoff = np.hstack((float(list(file_anno)[0]), file_anno[list(file_anno)[0]]))
        label_pos = np.where(tech_name == re.search('Iso_(.*).csv', anno_files).group(1))[0] + 1
        for n in range(len(file_onoff)//2):
            start_idx = int(file_onoff[2*n] * sr / hop_sample)  # use PET's hop_sample,alreay considered the feature duplication
            end_idx = int(file_onoff[2*n+1] * sr / hop_sample)
            if label_pos:
                label_id[k][label_pos-1, start_idx:end_idx] = np.ones((end_idx-start_idx), dtype=int) * (label_pos) # label position in tech_name[m] array
    elif wav_files[k].split('/')[2] == 'Piece': # 'Piece'
        for m in range(len(tech_name)):
            if os.path.exists(wav_files[k][:-4]+ '_tech_' + tech_name[m] + '.csv'):
                anno_files = (wav_files[k][:-4]+ '_tech_' + tech_name[m] + '.csv')
                file_anno = pd.read_csv(anno_files)
                file_onoff = np.hstack((float(list(file_anno)[0]), file_anno[list(file_anno)[0]]))
                for n in range(len(file_onoff)//2):
                    start_idx = int(file_onoff[2*n] * sr / hop_sample)
                    end_idx = int(file_onoff[2*n+1] * sr / hop_sample)
                    label_id[k][m, start_idx:end_idx] = np.ones((end_idx-start_idx), dtype=int) * (m+1)

# +
# use single-labeled part only
label_all = 0
import collections
for k in range(len(label_id)):
    # for m in range(label_id[k].shape[1]):  # no. time frame
    #     # if collections.Counter(label_id[k][:,m])[0] < 3:   # only one have label (counter=6)
    #     #     label_id[k][:,m] = np.ones((len(tech_name)),dtype=int) * 100
    label_all = np.hstack((label_all, np.sum(label_id[k],axis=0)))
    
label_id = label_all[1:]
del(label_all)

# +
# plt.plot(label_id)
# -

# Remove an outlier datapoint that contains all three PETs overlapping
player_id = np.delete(player_id, np.where(label_id==5), 0)
feature_conca = np.delete(feature_conca, np.where(label_id==5), 0)
file_id = np.delete(file_id, np.where(label_id==5), 0)
label_id = np.delete(label_id, np.where(label_id==5), 0)

print(label_id.shape, feature_conca.shape, player_id.shape, file_id.shape)

import collections
collections.Counter(label_id)

# # Playing technique recognition
# In the recognition process, the <a href="https://zenodo.org/record/3776864" title="CBFdataset">CBFdataset</a> is split into training and test sets according to an 8:2 ratio by performers (performers are randomly initialised).
# We conduct 5 splits in a circular way, with no performer overlap between the test sets across splits and between the training-test sets in each split.
# Within each split, we run a 3-fold cross-validation, sampling on the training dataset in a way that ensures each fold includes approximately the same ratio of positive and negative class instances for a given playing technique.
# This is to avoid the cases that there is no instance or there are too few instances of a given playing technique class in the validation set if we further split the training set based on performer identity. 

# for classification
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from thundersvm import SVC  # use GPU for SVM
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# classifier-SVM settings
kernel = 'rbf'; gpu_id = 0
# param_grid = {'C': [10], 'gamma': [.0001]}   # param_grid for toy experiment
param_grid = {'C': [256, 128, 64, 32, 16, 8], 'gamma': [2**(-12),2**(-11),2**(-10),2**(-9),2**(-8),2**(-7)]} # para_grid used
scoring = 'f1_macro'; cv = 3

# +
# data split according to players + cross validation
torch.manual_seed(42)
player_split = torch.randperm(len(np.unique(player_id))) + 1
player_split = player_split.numpy()

# trainSplit testSplit player
trainSplit = {k:[] for k in range(5)}; testSplit = {k:[] for k in range(5)}

trainSplit[0] = player_split[0:int(player_split.shape[0]*.8)]    # seg idx for trainSplit
testSplit[0] = player_split[int(player_split.shape[0]*.8):player_split.shape[0]]   # seg idx for testSplit

trainSplit[1] = player_split[2:10]    # seg idx for trainSplit
testSplit[1] = player_split[0:2]   # seg idx for testSplit

trainSplit[2] = np.hstack((player_split[4:10],player_split[0:2]))   # seg idx for trainSplit
testSplit[2] = player_split[2:4]   # seg idx for testSplit

trainSplit[3] = np.hstack((player_split[6:10],player_split[0:4]))  # seg idx for trainSplit
testSplit[3] = player_split[4:6]   # seg idx for testSplit

trainSplit[4] = np.hstack((player_split[8:10],player_split[0:6]))   # seg idx for trainSplit
testSplit[4] = player_split[6:8]   # seg idx for testSplit
# -

# record PRF and confusion obtained at each split
PRF = {split:np.zeros((len(tech_name)+1,3)) for split in range(5)} # including "other" class which is 0
confusion = {split:np.zeros((len(tech_name)+1, len(tech_name)+1)) for split in range(5)} 

# +
subset = np.ones((len(player_id)), dtype=int)*100

for k in range(len(player_id)):
    if player_id[k] in trainSplit[0]:
        subset[k] = 0
    else:
        subset[k] = 1

label_tr = label_id[subset == 0]
print(collections.Counter(label_tr))

# -

# ## 5 splits

# classification for each split
t0 = time.time()
for split in tqdm(range(5)):

    subset = np.ones((len(player_id)), dtype=int) * 100

    for k in range(len(player_id)):
        if player_id[k] in trainSplit[split]:
            subset[k] = 0
        else: # test
            subset[k] = 1

    feature_tr, label_tr = feature_conca[subset == 0], label_id[subset == 0]
    feature_te, label_te = feature_conca[subset == 1], label_id[subset == 1]

    #########################  imputation  ###############################
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    feature_tr = imp.fit_transform(feature_tr)
    feature_te = imp.transform(feature_te)

    #########################  normalisation  ###############################
    stdscaler = StandardScaler()
    feature_tr = stdscaler.fit_transform(feature_tr)
    feature_te = stdscaler.transform(feature_te)
    print(feature_tr.shape, feature_te.shape)

    #########################  classification  ###############################
    clf =  GridSearchCV(SVC(kernel=kernel, gpu_id=gpu_id), param_grid=param_grid, cv=cv, scoring=scoring)
    clf = clf.fit(feature_tr, label_tr)
    label_pred = clf.predict(feature_te)
    print('Result of split %d :' % split)
    print(classification_report(label_te, label_pred))
    print(confusion_matrix(label_te, label_pred))
    
    #########################  record result of each split  ###############################
    # extract P,R,F values from classification_report
    lineSep = 54 ; dist = 10; Pos_firstNum = classification_report(label_te, label_pred).find('\n') + 21 
    for k in range(len(tech_name)+1):
        PRF[split][k,:] = np.array([
        float(classification_report(label_te, label_pred)[Pos_firstNum+lineSep*k:Pos_firstNum+4+lineSep*k]),\
        float(classification_report(label_te, label_pred)[Pos_firstNum+dist*1+lineSep*k:Pos_firstNum+4+dist*1+lineSep*k]),\
        float(classification_report(label_te, label_pred)[Pos_firstNum+dist*2+lineSep*k:Pos_firstNum+4+dist*2+lineSep*k])])
    confusion[split] = confusion_matrix(label_te, label_pred)

print('Classifcation takes %.2f hours.' %((time.time() - t0)/3600))

np.savez('CBFdataset_PRF_confusion.npz', PRF, confusion)

# ## average

PRF_aver = np.mean(np.array([PRF[k] for k in range(5)]), 0)
print('F-measure for each type of playing technique: ')
print((PRF_aver[:,2]))
print('Marco F-measure: %.3f'%np.mean(PRF_aver[:,2]))

confusion_sum = np.sum(np.array([confusion[k] for k in range(5)]), 0)
print('Confusion matrix on the CBFdataset:')
print(confusion_sum)

# ## confusion matrix

# +
A = confusion_sum
B = np.zeros((confusion_sum.shape[0]+1, confusion_sum.shape[0]+1), dtype=int)
B[:-1, :-1] = A
B[-1, :] = B [0, :]; B[:, -1] = B [:, 0]
B = B[1:, 1:]
confusion = B

tech_name = ['tremolo', 'acciaccatura', 'glissando', 'trill', 'flutter-tongue', 'vibrato', 'portamento']
tech_name.append('other')
tech_name = np.array(tech_name)
# -

norm_confusion = confusion.T / confusion.astype(np.float).sum(axis=1)
norm_confusion = norm_confusion.T

# +
################################# without adapt duplicate & hopsample/2 because of multiplier in joint cal ####################
# use seaborn plotting defaults
import seaborn as sns; sns.set()
from matplotlib import rcParams

plt.figure(figsize=(16,6))
plt.subplot(121)
sns.heatmap(confusion, cmap = "Blues", square=True, annot=True, fmt="d",
            xticklabels=tech_name, yticklabels=tech_name)
plt.xticks([0,1,2,3,4,5,6,7], tech_name, rotation=60, fontsize=11.5); plt.yticks(fontsize=11.5)
plt.ylabel('True label', fontsize=12); plt.xlabel('Predicted label', fontsize=12)
plt.ylim([8,0])
plt.title('(a) Confusion', fontsize=13)
rcParams['axes.titlepad'] = 15

plt.subplot(122)
norm_confusion = np.round(norm_confusion,2)
sns.heatmap(norm_confusion, cmap = "Blues", square=True, annot=True,
            xticklabels=tech_name, yticklabels=tech_name) # cbar=False,
plt.xticks([0,1,2,3,4,5,6,7], tech_name, rotation=60, fontsize=11.5); plt.yticks(fontsize=11.5)
plt.ylabel('True label', fontsize=12); plt.xlabel('Predicted label', fontsize=12)
plt.ylim([8,0])
plt.title('(b) Normalised confusion', fontsize=13)
rcParams['axes.titlepad'] = 15

plt.tight_layout()
