import numpy as np
import librosa
import plotly.graph_objects as go

# RAMKA
def compute_volume(frame):
    """
    Oblicza głośność (volume) dla pojedynczej ramki.
    """
    volume = np.sqrt(np.mean(frame**2)) 
    return volume

def compute_ste(frame):
    """
    Oblicza energię krótkoczasową (STE) dla pojedynczej ramki.
    """
    volume = compute_volume(frame)
    return volume ** 2

def compute_zcr(frame):
    """
    Oblicza stopę przejść przez zero (Zero-Crossing Rate) dla pojedynczej ramki.
    """
    zcr_value = np.count_nonzero(np.diff(np.sign(frame))) / len(frame)
    return zcr_value


def compute_amdf(frame):
    """Oblicza AMDF, normalizując wartości."""
    frame_length = len(frame)
    amdf_values = np.zeros(frame_length)

    for lag in range(1, frame_length):
        amdf_values[lag] = np.sum(np.abs(frame[:-lag] - frame[lag:])) 
    
        amdf_values[lag] = amdf_values[lag] / (frame_length - lag)

    return amdf_values

def compute_autocorrelation(frame):
    """Oblicza autokorelację, normalizując wartości."""
    frame_length = len(frame)
    autocorr_values = np.zeros(frame_length)

    for lag in range(frame_length):
        autocorr_values[lag] = np.sum(frame[:frame_length - lag] * frame[lag:])  

    if autocorr_values[0] > 0:
        autocorr_values /= autocorr_values[0]

    return autocorr_values


def compute_f0(frame, sr, method="amdf"):
    """
    Obliczanie częstotliwości podstawowej F0 za pomocą AMDF lub autokorelacji,
    uwzględniając realistyczne zakresy dla mowy.
    """
    min_lag = sr // 400  # Maksymalna częstotliwość mowy = 400 Hz
    max_lag = sr // 50   # Minimalna częstotliwość mowy = 50 Hz

    if method == "amdf":
        values = compute_amdf(frame)
        # values = values / np.max(values) 
        
        if min_lag < max_lag:
            valid_range = values[min_lag:max_lag]
            if valid_range.size > 0:
                min_lag = np.argmin(valid_range) + min_lag
            else:
                min_lag = 0
        else:
            min_lag = 0

    elif method == "autocorrelation":
        values = compute_autocorrelation(frame)
        
        if min_lag < max_lag:
            valid_range = values[min_lag:max_lag]
            if valid_range.size > 0:
                min_lag = np.argmax(valid_range) + min_lag
            else:
                min_lag = 0
        else:
            min_lag = 0

    else:
        raise ValueError("Nieznana metoda. Wybierz 'amdf' lub 'autocorrelation'.")
    

    f0 = sr / min_lag if min_lag > 0 else 0
    return f0

# KLIP
def compute_vstd(volumes):
    """
    Oblicza odchylenie standardowe głośności dla danego klipu.
    """
    return np.std(volumes) / np.max(volumes)

def compute_vdr(volumes):
    """
    Oblicza dynamiczny zakres głośności (VDR) dla danego klipu.
    """
    return (np.max(volumes) - np.min(volumes)) / np.max(volumes)

def compute_vu(volumes):
    """
    Oblicza różnicę między kolejnymi wartościami głośności.
    """
    if len(volumes) < 2:
        return 0  
    return np.sum(np.abs(np.diff(volumes)))

def compute_lster(ste, sr, hop_length):
    """
    Oblicza Low Short-Time Energy Ratio (LSTER) dla danego klipu.
    """
    N = len(ste)  

    avg_ste = np.mean(ste)  

    lster_value = np.sum(ste < (0.5 * avg_ste)) / N
    
    return lster_value

def compute_energy_entropy(clip, segment_size=10):
    """
    Oblicza entropię energii dla danego klipu.
    """
    if len(clip) == 0:
        return 0  

    total_energy = np.sum(clip ** 2) 

    if total_energy == 0:
        return 0

    J = int(np.ceil(len(clip) / segment_size)) 
    entropy = 0  

    for i in range(J):
        start = i * segment_size
        end = min(start + segment_size, len(clip))  
        segment = clip[start:end]  
        segment_energy = np.sum(segment ** 2) 

        normalized_energy = segment_energy / total_energy

        if normalized_energy > 0:
            entropy += normalized_energy * np.log2(normalized_energy)

    return -entropy


def compute_zstd(clip_zcr):
    """
    Oblicza odchylenie standardowe ZCR dla danego klipu
    """
    if len(clip_zcr) > 0:
        return np.std(clip_zcr)
    else:
        return 0  
    

def compute_hzcrr(clip_zcr):
    """
    Oblicza High Zero-Crossing Rate Ratio (HZCRR) dla danego klipu
    """
    if len(clip_zcr) > 0:
        avg_zcr = np.mean(clip_zcr)  
        high_zcr_count = np.sum(clip_zcr > avg_zcr) 
        hzcrr = high_zcr_count / len(clip_zcr)
        return hzcrr
    else:
        return 0 


def extract_audio_features(y, sr, hop_length, frame_size_ms=20, clip_size_s=1000,
                            music_lster_threshold = 0.5, music_zstd_threshold = 0.15, 
                            voicing_ste_threshold=0.0005, voicing_volume_threshold=0.005, 
                            silence_zcr_threshold=0.05, silence_volume_threshold=0.001, method='amdf'):
    """
    Ekstrakcja cech dla ramek i klipów z sygnału audio.
    """
    frame_size = int(sr * frame_size_ms / 1000) 
    clip_size = min(len(y), int(sr * clip_size_s / 1000)) 

    frames = [y[i:min(i + frame_size, len(y))] for i in range(0, len(y), frame_size)]

    clip_features = {
        'vstd': [],
        'vdr': [],
        'vu': [],
        'lster': [],
        'energy_entropy': [],
        'zstd': [],
        'hzcrr': [],
        'is_music': []
    }

    frame_features = {
        'volume': [],
        'ste': [],
        'zcr': [],
        'f0': [],
        'is_silence': [],
        'is_voiced': []
    }

    for frame in frames:
        volume = compute_volume(frame)
        ste = compute_ste(frame)
        zcr = compute_zcr(frame)
        f0 = compute_f0(frame, sr, method)
        is_silence = detect_silence(zcr, volume, silence_zcr_threshold, silence_volume_threshold)
        is_voiced = classify_voicing(ste, volume, voicing_ste_threshold, voicing_volume_threshold)

        frame_features['volume'].append(volume)
        frame_features['ste'].append(ste) 
        frame_features['zcr'].append(zcr)
        frame_features['f0'].append(f0)
        frame_features['is_silence'].append(is_silence)
        frame_features['is_voiced'].append(is_voiced)

    for i in range(len(frame_features['f0'])):
        if not frame_features['is_voiced'][i]:
            frame_features['f0'][i] = 0
    
    num_clips = (len(y) + clip_size - 1) // clip_size

    for i in range(num_clips):
        start_idx = i * clip_size // frame_size
        end_idx = min((i + 1) * clip_size // frame_size, len(frame_features["volume"]))

        clip = y[start_idx:end_idx]

        volumes_in_clip = [frame_features['volume'][i] for i in range(start_idx, end_idx)]
        ste_in_clip = [frame_features['ste'][i] for i in range(start_idx, end_idx)]
        zcr_in_clip = [frame_features['zcr'][i] for i in range(start_idx, end_idx)]

        if len(volumes_in_clip) == 0 or len(ste_in_clip) == 0 or len(zcr_in_clip) == 0:
            continue

        clip_features['vstd'].append(compute_vstd(volumes_in_clip)) 
        clip_features['vdr'].append(compute_vdr(volumes_in_clip)) 
        clip_features['vu'].append(compute_vu(volumes_in_clip)) 
        clip_features['lster'].append(compute_lster(ste_in_clip, sr, hop_length))  
        clip_features['energy_entropy'].append(compute_energy_entropy(clip)) 
        clip_features['zstd'].append(compute_zstd(zcr_in_clip))  
        clip_features['hzcrr'].append(compute_hzcrr(zcr_in_clip)) 

        is_music = detect_music(clip_features['lster'][i], clip_features['zstd'][i], music_lster_threshold, music_zstd_threshold)
        clip_features['is_music'].append(is_music)

    return frame_features, clip_features


def plot_feature(y, sr, feature_values, feature_name, mode):
    """
    Tworzy interaktywny wykres dla wartości cechy w czasie.
    """
    time_axis = np.linspace(0, len(y) / sr, num=len(feature_values))

    frame_numbers = np.arange(1, len(feature_values) + 1)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=time_axis, 
        y=feature_values, 
        mode='lines', 
        name=feature_name,
        hovertemplate=
            f"<b>Numer {mode}:</b> %{{customdata[0]}}<br>" +
            "<b>Czas:</b> %{x:.3f} s" + "<br>" +
            f"<b>{feature_name}:</b> %{{y:.3f}}" + "<extra></extra>", 
        customdata=np.array([frame_numbers]).T  
    ))

    fig.update_layout(
        title=f"{feature_name}",
        title_x=0.5,
        xaxis_title="Czas (s)",
        yaxis_title=f"{feature_name}",
        hovermode="closest" 
    )

    return fig


def detect_silence(zcr, volume, zcr_threshold=0.01, volume_threshold=0.005):
    """
    Detekcja fragmentów ciszy na podstawie wartości ZCR i głośności.
    """
    if zcr > zcr_threshold and volume < volume_threshold:
        return True # Cisza
    else:
        return False # Brak ciszy

def classify_voicing(ste_value, volume_value, threshold_ste=0.0005, threshold_volume=0.005):
    """
    Klasyfikacja fragmentu jako dźwięczny lub bezdźwięczny.
    """
    if ste_value > threshold_ste and volume_value > threshold_volume:
        return True  # Dźwięczny
    else:
        return False  # Bezdźwięczny


def detect_music(lster, zstd, lster_threshold = 0.5, zstd_threshold = 0.15):
    if lster > lster_threshold and zstd < zstd_threshold:
        return False  # Mowa
    else:
        return True  # Muzyka
