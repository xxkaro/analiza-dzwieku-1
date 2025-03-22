import numpy as np
import librosa


def compute_volume(y, frame_size, hop_length):
    volumes = []
    for start in range(0, len(y), hop_length):
        end = min(start + frame_size, len(y))
        frame = y[start:end]
        volume = np.sqrt(np.mean(frame**2))  
        volumes.append(volume)
    return np.array(volumes)

def compute_ste(y, frame_size, hop_length):
    return compute_volume(y, frame_size, hop_length) ** 2

def compute_zcr(y, frame_size, hop_length):
    zcr = []
    for start in range(0, len(y), hop_length):
        end = min(start + frame_size, len(y))
        frame = y[start:end]
        zcr_value = np.count_nonzero(np.diff(np.sign(frame))) / len(frame)
        zcr.append(zcr_value)
    return np.array(zcr)

def compute_auto_corr(frame):
    N = len(frame) 
    auto_corr = np.zeros(N)  

    for l in range(N):
        auto_corr[l] = np.sum(frame[:N-l] * frame[l:N]) 
    return auto_corr

def compute_amdf(frame):
    N = len(frame)  
    amdf = np.zeros(N)  
    for l in range(N):
        amdf[l] = np.sum(np.abs(frame[:N-l] - frame[l:N])) # / (N - l) 
    return amdf

def find_f0(frame, sr, method='auto_corr'):
    if method == 'auto_corr':
        f0 = compute_auto_corr(frame)
        max_idx = np.argmax(f0[1:]) + 1
        if max_idx == 0:
            return 0
        f0 = sr / max_idx
        
    elif method == 'amdf':
        amdf = compute_amdf(frame)
        min_idx = np.argmin(amdf[1:]) + 1
        if min_idx == 0:
            return 0 
        f0 = sr / min_idx
    
    return f0


def compute_vstd(volumes):
    return np.std(volumes) / np.max(volumes)

def compute_vdr(volumes):
    return np.max(volumes) - np.min(volumes) / np.max(volumes)

def compute_vu(volumes):
    if len(volumes) < 2:
        return 0  
    vu = np.sum(np.abs(np.diff(volumes))) 
    return vu

def compute_lster(ste, sr, hop_length):
    N = len(ste)  
    frames_per_sec = sr // hop_length  

    lster_values = np.zeros(N)
    
    for i in range(N):
        # Okno 1-sekundowe (symetryczne wokół ramki i)
        start = max(0, i - frames_per_sec // 2)
        end = min(N, i + frames_per_sec // 2)
        
        avg_ste = np.mean(ste[start:end])  
  
        lster_values[i] = np.sum(ste[start:end] < 0.5 * avg_ste)
    
    return np.mean(lster_values) / 2


def compute_energy_entropy(y, segment_size=1024):
    N = len(y)  
    total_energy = np.sum(y ** 2) 
    J = N // segment_size
    entropy = 0

    for i in range(J):
        start = i * segment_size  
        end = start + segment_size
        segment = y[start:end]  

        segment_energy = np.sum(segment ** 2)

        normalized_energy = segment_energy / total_energy

        if normalized_energy > 0:
            entropy += normalized_energy * np.log2(normalized_energy)

    energy_entropy = -entropy
    return energy_entropy

def compute_zstd(zcr, clip_length, sr, hop_length):
    frames_per_sec = sr // hop_length 
    clip_frames = clip_length * frames_per_sec 
    
    zstd_values = []
    
    for i in range(0, len(zcr), clip_frames):
        clip_zcr = zcr[i:i + clip_frames] 
        
        if len(clip_zcr) > 0:
            zstd_value = np.std(clip_zcr)  
            zstd_values.append(zstd_value)
    
    return zstd_values

def compute_hzcrr(zcr, clip_length, sr, hop_length):
    frames_per_sec = sr // hop_length  
    clip_frames = clip_length * frames_per_sec  
    
    hzcrr_values = []
    
    for i in range(0, len(zcr), clip_frames):
        clip_zcr = zcr[i:i + clip_frames] 
        
        if len(clip_zcr) > 0:
            avg_zcr = np.mean(clip_zcr)  
            high_zcr_count = np.sum(clip_zcr > avg_zcr) 
            hzcrr_value = high_zcr_count / len(clip_zcr)
            hzcrr_values.append(hzcrr_value)
    
    return hzcrr_values


def plot_feature(y, sr, feature_values, feature_name, frame_size, hop_length):
    import plotly.graph_objects as go
    
    time_axis = np.linspace(0, len(y) / sr, num=len(feature_values))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_axis, y=feature_values, mode='lines', name=feature_name))

    fig.update_layout(
        title=f"Feature: {feature_name}",
        xaxis_title="Czas (s)",
        yaxis_title=f"Wartość {feature_name}",
        hovermode="closest"
    )

    return fig

def detect_silence(y, frame_size, hop_length, silence_threshold=0.005, zcr_threshold=0.1):
    volumes = compute_volume(y, frame_size, hop_length)
    zcr = compute_zcr(y, frame_size, hop_length)
    silence_flags = (volumes < silence_threshold) # & (zcr < zcr_threshold)

    return silence_flags.tolist()

def detect_music_speech(y, frame_size, hop_length, lster_threshold=0.1, zstd_threshold=0.1):
    zcr = compute_zcr(y, frame_size, hop_length)

    ste = compute_ste(y, frame_size, hop_length)

    lster = compute_lster(ste, librosa.get_samplerate(y), hop_length)

    zstd = compute_zstd(zcr, clip_length=1, sr=librosa.get_samplerate(y), hop_length=hop_length)

    music_speech_flags = []

    for i in range(len(zcr)):
        if lster < lster_threshold:
            music_speech_flags.append(0)  
        elif zstd[i] > zstd_threshold:
            music_speech_flags.append(1)  
        else:
            music_speech_flags.append(-1) 

    return music_speech_flags


    