"""
feature_extractor.py
--------------------
Standardized audio feature extraction for the GenreMap system.

Extracts exactly 57 features in a consistent order, matching the GTZAN dataset format.
This ensures uploaded songs are compatible with the trained dimensionality reduction models.

Features extracted (in order):
  - Chroma STFT (mean, var)
  - RMS (mean, var)
  - Spectral centroid (mean, var)
  - Spectral bandwidth (mean, var)
  - Spectral rolloff (mean, var)
  - Zero crossing rate (mean, var)
  - Harmony (mean, var)
  - Perceptr (mean, var)
  - Tempo
  - MFCCs 1-20 (mean, var for each)

Usage:
  from feature_extractor import extract_features
  
  features = extract_features('my_song.wav')
  # Returns list of 57 floats in standard order
"""

import numpy as np
import librosa


# This is the canonical feature order - DO NOT CHANGE
FEATURE_ORDER = [
    'chroma_stft_mean', 'chroma_stft_var',
    'rms_mean', 'rms_var',
    'spectral_centroid_mean', 'spectral_centroid_var',
    'spectral_bandwidth_mean', 'spectral_bandwidth_var',
    'rolloff_mean', 'rolloff_var',
    'zero_crossing_rate_mean', 'zero_crossing_rate_var',
    'harmony_mean', 'harmony_var',
    'perceptr_mean', 'perceptr_var',
    'tempo',
] + [f'mfcc{i}_{stat}' for i in range(1, 21) for stat in ['mean', 'var']]


def extract_features(audio_path: str, duration: float = 30.0) -> list[float]:
    """
    Extract standardized audio features from a WAV file.
    
    Args:
        audio_path: Path to audio file (WAV, MP3, etc.)
        duration: Length to analyze in seconds (default: 30s to match GTZAN)
    
    Returns:
        List of 57 floats in the canonical feature order
    
    Raises:
        Exception: If audio file cannot be loaded or features cannot be extracted
    """
    try:
        # Load audio - force mono and resample
        y, sr = librosa.load(audio_path, duration=duration, sr=22050, mono=True)
        
        if len(y) == 0:
            raise ValueError("Audio file is empty or too short")
        
        # Ensure minimum length (at least 1 second)
        if len(y) < sr:
            raise ValueError(f"Audio too short: {len(y)/sr:.2f}s (need at least 1s)")
        
        # Extract features
        features = {}
        
        # Chroma STFT
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_stft_mean'] = float(np.mean(chroma_stft))
        features['chroma_stft_var'] = float(np.var(chroma_stft))
        
        # RMS Energy
        rms = librosa.feature.rms(y=y)
        features['rms_mean'] = float(np.mean(rms))
        features['rms_var'] = float(np.var(rms))
        
        # Spectral Centroid
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid_mean'] = float(np.mean(spec_cent))
        features['spectral_centroid_var'] = float(np.var(spec_cent))
        
        # Spectral Bandwidth
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features['spectral_bandwidth_mean'] = float(np.mean(spec_bw))
        features['spectral_bandwidth_var'] = float(np.var(spec_bw))
        
        # Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features['rolloff_mean'] = float(np.mean(rolloff))
        features['rolloff_var'] = float(np.var(rolloff))
        
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zero_crossing_rate_mean'] = float(np.mean(zcr))
        features['zero_crossing_rate_var'] = float(np.var(zcr))
        
        # Harmonic and Percussive
        y_harm, y_perc = librosa.effects.hpss(y)
        features['harmony_mean'] = float(np.mean(y_harm))
        features['harmony_var'] = float(np.var(y_harm))
        features['perceptr_mean'] = float(np.mean(y_perc))
        features['perceptr_var'] = float(np.var(y_perc))
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        # Handle case where tempo might be an array
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
        else:
            tempo = float(tempo)
        features['tempo'] = tempo
        
        # MFCCs (1-20)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(20):
            features[f'mfcc{i+1}_mean'] = float(np.mean(mfccs[i]))
            features[f'mfcc{i+1}_var'] = float(np.var(mfccs[i]))
        
        # Return in canonical order
        feature_list = [float(features[name]) for name in FEATURE_ORDER]
        
        # Validate
        assert len(feature_list) == 57, f"Expected 57 features, got {len(feature_list)}"
        assert all(np.isfinite(f) for f in feature_list), "Features contain NaN or inf"
        
        return feature_list
        
    except Exception as e:
        raise Exception(f"Feature extraction failed for {audio_path}: {str(e)}")


def features_to_dict(feature_list: list[float]) -> dict[str, float]:
    """
    Convert feature list to named dictionary.
    
    Args:
        feature_list: List of 57 features in canonical order
    
    Returns:
        Dictionary mapping feature names to values
    """
    if len(feature_list) != 57:
        raise ValueError(f"Expected 57 features, got {len(feature_list)}")
    
    return {name: val for name, val in zip(FEATURE_ORDER, feature_list)}


def get_feature_names() -> list[str]:
    """
    Get the canonical list of feature names in order.
    
    Returns:
        List of 57 feature names
    """
    return FEATURE_ORDER.copy()


# ── Testing ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python feature_extractor.py <audio_file.wav>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    print(f"Extracting features from: {audio_file}")
    try:
        features = extract_features(audio_file)
        feature_dict = features_to_dict(features)
        
        print(f"\n✓ Extracted {len(features)} features")
        print("\nSample features:")
        for name, val in list(feature_dict.items())[:10]:
            print(f"  {name:30s} {val:.6f}")
        print("  ...")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
