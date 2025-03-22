import matplotlib.pyplot as plt
import librosa.display
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np

def plot_waveform(y, sr, highlight_regions=None):
    """
    Funkcja rysująca wykres przebiegu czasowego audio, z opcjonalnym naniesieniem zaznaczonych regionów.
    
    highlight_regions: Lista krotek (start, stop) - regiony do zaznaczenia na wykresie
    """
    # Czas na osi X
    time_axis = np.linspace(0, len(y) / sr, num=len(y))
    
    # Tworzymy wykres audio
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_axis, y=y, mode='lines', name="Sygnał"))

    # Jeżeli są regiony do zaznaczenia
    if highlight_regions:
        for region in highlight_regions:
            fig.add_vrect(
                x0=region[0], x1=region[1],
                fillcolor="rgba(255, 0, 0, 0.3)", 
                line=dict(color="red", width=1),
                annotation_text="Cisza" if len(region) == 2 else "Mowa/Muzyka",
                annotation_position="top left"
            )

    # Ustawienia wykresu
    fig.update_layout(
        title="Przebieg czasowy",
        xaxis_title="Czas (s)",
        yaxis_title="Amplituda",
        hovermode="closest",  # Zbliżenie na najbliższą próbkę
    )

    return fig



def plot_autocorrelation(autocorr, amdf):
    """Rysuje wykresy autokorelacji i AMDF."""
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    
    ax[0].plot(autocorr)
    ax[0].set_title("Autokorelacja")
    ax[0].set_xlabel("Próbki")
    
    ax[1].plot(amdf)
    ax[1].set_title("AMDF")
    ax[1].set_xlabel("Próbki")
    
    st.pyplot(fig)

def save_results(params, filename):
    """Zapisuje parametry audio do pliku CSV."""
    df = pd.DataFrame([params])
    df.to_csv(filename, index=False)


def plot_interactive_waveform(y, sr, frame_duration):
    """Generuje interaktywny wykres z możliwością wyświetlania danych w hover info"""
    
    # Czas na osi X
    time_axis = np.linspace(0, len(y) / sr, num=len(y))
    
    # Parametry do ramki
    frame_size = int((frame_duration / 1000) * sr)
    
    # Tworzymy wykres audio
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_axis, y=y, mode='lines', name="Sygnał"))

    # Dodajemy etykiety na hover
    hover_data = []
    for i in range(0, len(y), frame_size):
        frame = y[i:i + frame_size]
        avg_amplitude = np.mean(np.abs(frame))  # Przykładowy parametr: średnia amplituda
        hover_data.append(f"Avg Amplitude: {avg_amplitude:.3f}")

    fig.update_traces(
        hoverinfo="text",  # Wyświetlamy tylko tekst w hoverze
        hovertext=hover_data  # Dane, które mają się pojawić w hoverze
    )

    # Ustawienia wykresu
    fig.update_layout(
        title="Przebieg czasowy",
        xaxis_title="Czas (s)",
        yaxis_title="Amplituda",
        hovermode="closest",  # Zbliżenie na najbliższą próbkę
    )

    return fig