import os
import numpy as np
import soundfile as sf
import librosa
import pandas as pd
from shlex import quote
import math
import essentia.standard as es



def extractPitch(filename, fs, overwrite = False, vampdir='../'):
    
    filename_sansExt = os.path.splitext(filename)[0]
    pitchFilename = f"{filename_sansExt}_vamp_pyin_pyin_smoothedpitchtrack.lab"

    if os.path.exists(pitchFilename) == False or overwrite == True:
        data, samplerate = sf.read(filename)

#         print(samplerate)
        if samplerate != fs:
            cmd = f"ffmpeg -i {quote(filename)} -ar {fs} {quote(filename_sansExt)}_temp.wav"
            os.system(cmd)
            os.rename(f"{filename_sansExt}_temp.wav", filename)

        cmd = f"./{vampdir}sonic-annotator -t pyin_params.n3 {quote(filename)} -w lab --lab-force"
#         print(os.popen(cmd).read())
        os.system(cmd)
        
    pitch_lab = np.loadtxt(pitchFilename, delimiter='\t')
    pitch = pitch_lab[:,1].astype(np.float32)
    pitch[np.where(pitch < 0)[0]] *= 0
    time = pitch_lab[:,0].astype(np.float32)
    # os.remove(pitchFilename)
    
    return pitch, time


def hz2midi(pitch):
    if type(pitch) == float:
        pitch = 12.0 * np.log2((pitch + 1e-7)/440.0) + 69.0
        if pitch < 0.0:
            pitch = 0.0
    elif type(pitch) == np.ndarray:
        pitch[np.where(pitch == 0.0)] = 1e-7
        pitch = 12.0 * np.log2(pitch/440.0) + 69.0
        pitch[np.where(pitch < 0.0)] = 0.0
    else:
        raise Exception("Expected type: float or numpy.ndarray")
    return pitch

def nearest(arr, val):
    d = (np.abs(arr - val)).argmin()
    return d

def pitchMedianFilter(pitch, kernelSize=5):
    medianFilter = es.MedianFilter(kernelSize=kernelSize)
    pitch = medianFilter(pitch) # smoothen and remove octave jumps
    return pitch

def tokenize_portamento(csv_path, dPitch, t_pitch, rms, feature_type, token_time=100, hop_time=20, remove_nan=True):
    pitch_sr = len(t_pitch)/(t_pitch[-1] - t_pitch[0])
    df = pd.read_csv(csv_path, header=None)
    y = []
    for t in t_pitch:
        target_times = np.array(df[0])
        targets = np.array(df[1])
        nearest_ix = nearest(target_times,t)
        if t < target_times[0] or t > target_times[-1]:
            y.append(0)

        elif t >= target_times[nearest_ix]:
            if targets[nearest_ix] == f'on_{feature_type}':
                y.append(1)
            else:
                y.append(0)
        else:
            if targets[nearest_ix-1] == f'on_{feature_type}':
                y.append(1)
            else:
                y.append(0)

    token_size = int(token_time*0.001*pitch_sr)
    # hop_size = int(20*0.001*pitch_sr)
    if hop_time is None:
        hop_size = int(token_size/2)
    hop_size = int(hop_time*0.001*pitch_sr)
    target_list = []
    tokens = []
    for ix in range(0, len(t_pitch), token_size-hop_size):

        dPitch_token = list(dPitch[ix:ix+token_size])
        rms_token = rms[ix:ix+token_size]
        if remove_nan == True and sum(dPitch_token) < token_size/2 * -100:
            continue
        tokens.append(np.array([dPitch_token, rms_token]).T)
        if sum(y[ix:ix+token_size]) >= 0.8*token_size:
            target_list.append(1)
        else:
            target_list.append(0)
            
    return tokens, target_list

def tokenize_glissando(csv_path, pitch, t_pitch, rms, feature_type, token_time=200, hop_time=20, remove_nan=True):
    pitch_sr = len(t_pitch)/(t_pitch[-1] - t_pitch[0])
    df = pd.read_csv(csv_path, header=None)
    y = []
    for t in t_pitch:
        target_times = np.array(df[0])
        targets = np.array(df[1])
        nearest_ix = nearest(target_times,t)
        if t < target_times[0] or t > target_times[-1]:
            y.append(0)

        elif t >= target_times[nearest_ix]:
            if targets[nearest_ix] == f'on_{feature_type}':
                y.append(1)
            else:
                y.append(0)
        else:
            if targets[nearest_ix-1] == f'on_{feature_type}':
                y.append(1)
            else:
                y.append(0)

    token_size = int(token_time*0.001*pitch_sr)
    # hop_size = int(20*0.001*pitch_sr)
    if hop_time is None:
        hop_size = int(token_size/2)
    hop_size = int(hop_time*0.001*pitch_sr)
    target_list = []
    tokens = []
    prev_pitch = np.zeros(token_size)
    prev_rms = np.zeros(token_size)
    
    for ix in range(0, len(t_pitch), token_size-hop_size):

        if len(tokens) != 0:    
            dPitch = pitch[ix:ix+token_size] - prev_pitch[0:min(token_size,len(pitch[ix:ix+token_size]))]
            dPitch_token = dPitch/token_size
               
            dRms = rms[ix:ix+token_size] - prev_rms[0:min(token_size,len(rms[ix:ix+token_size]))]
            dRms_token = dRms/token_size
            
        else:
            dPitch_token = prev_pitch
            dRms_token = prev_rms
            
        rms_token = rms[ix:ix+token_size]/token_size
        
        dPitch_token[np.isnan(dPitch_token)] = -100
        dRms_token[np.isnan(dRms_token)] = -100
        rms_token[np.isnan(rms_token)] = -100
        
        prev_pitch = pitch[ix:ix+token_size]
        prev_rms = rms[ix:ix+token_size]
        
        if sum(dPitch_token) < token_size/4 * -100 or sum(rms_token) < token_size/4 * -100 :
            continue
        tokens.append(np.array([dPitch_token, dRms_token]).T)
        if sum(y[ix:ix+token_size]) >= 0.9*token_size:
            target_list.append(1)
        else:
            target_list.append(0)
            
    return tokens, target_list

def extract_features(fpath, feature_type):
    pitch,t_pitch = extractPitch(fpath, fs=44100, overwrite = False, vampdir='../')
    pitch[pitch == 0] = np.nan
    pitch = hz2midi(pitch)
    y, sr = librosa.load(fpath,sr=44100)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=256, center=True)[0]

    
    fname = fpath.split('/')[-1]
    try:
        f_dir = '/'.join(fpath.split('/')[:-1])
        target_csv = [f for f in os.listdir(f_dir) if (
            f.endswith('.csv') and fname.split('.wav')[0] in f)][0]
    except IndexError:
        return None, None
    csv_path = os.path.join(f_dir, target_csv)
    
    if feature_type == 'Portamento':
        dPitch = pitch - np.concatenate((np.array([0]),pitch),axis=0)[:-1]
        dPitch = np.array([-100 if math.isnan(p) else p for p in dPitch])
        rms, dPitch = sync_features(rms,dPitch)
        tokens, targets = tokenize_portamento(
            csv_path, 
            dPitch, 
            t_pitch, 
            rms, 
            feature_type=feature_type)

    elif feature_type == 'Glissando':
        pitch = np.round(pitchMedianFilter(pitch, kernelSize=5))
        rms = librosa.amplitude_to_db(rms)
        rms[rms<=-40] = np.nan
        rms, dPitch = sync_features(rms,pitch)
        tokens, targets = tokenize_glissando(
            csv_path, 
            pitch, 
            t_pitch, 
            rms, 
            feature_type=feature_type)
    
    
#     return np.array(tokens), np.array(targets)
    return tokens, targets

def sync_features(rms, dPitch):
    if len(rms) > len(dPitch):
        d = len(rms) - len(dPitch)
        rms = rms[min(d,3):len(rms)-max(0,(d - min(d,3)))]
    elif len(rms) < len(dPitch):
        d = len(dPitch) - len(rms)
        dPitch = dPitch[int(d/2)+1:len(dPitch)-int(d/2)]
    else:
        pass
    return rms, dPitch

def extract_candidates(pitch, pitch_sr):

    segments = []
    d = 0
    seg = []
    for ix,p in enumerate(pitch[:-1]):
        # Positive direction
        if np.isnan(p) or pitch[ix+1] - p < 0:
            if d >= 4:
                segments.append(seg)
            seg = []
            d=0     
        elif pitch[ix+1] - p > 0:
            d += 1
            seg.append(ix)
        elif pitch[ix+1] - p == 0:
            seg.append(ix)

    d = 0
    seg = []
    for ix,p in enumerate(pitch[:-1]):
        # Negative direction
        if np.isnan(p) or pitch[ix+1] - p > 0:
            if d >= 4:
                segments.append(seg)
            seg = []
            d=0     
        elif pitch[ix+1] - p < 0:
            d += 1
            seg.append(ix)
        elif pitch[ix+1] - p == 0:
            seg.append(ix)

    pruned_segments = []
    for seg in segments:
        if len(seg) >= 0.2 * pitch_sr:
            pruned_segments.append(seg)
            
    pruned_segments = [item for sublist in pruned_segments for item in sublist]

    cdts = []
    for ix in range(len(pitch)):
        if ix in pruned_segments:
            cdts.append(1)      
        else:
            cdts.append(0)

    return cdts

