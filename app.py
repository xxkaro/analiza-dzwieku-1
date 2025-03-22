import streamlit as st
import librosa
import numpy as np
import plotly.graph_objects as go
from audio_analysis import (compute_volume, compute_ste, compute_zcr, 
                            compute_vstd, compute_vdr, compute_lster, 
                            find_f0, compute_energy_entropy, compute_zstd, compute_hzcrr, 
                            plot_feature, detect_silence, detect_music_speech)

st.markdown(
    """
    <style>
        .main {
            max-width: 90% !important;
        }
        .block-container {
            padding-top: 2rem;
            padding-right: 10;
            padding-left: 10;
            max-width: 90% !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
# Tytuł aplikacji
st.title("🎵 Analiza sygnału audio")

# Wczytanie pliku audio
uploaded_file = st.file_uploader("Wybierz plik WAV", type=["wav"])


# Przygotowanie zmiennej dla clip_duration
clip_duration = None  # Definiujemy clip_duration poza warunkami

if uploaded_file is not None:
    # Dodanie opcji odtwarzania pliku audio
    st.audio(uploaded_file)
    # Wczytanie pliku audio
    y, sr = librosa.load(uploaded_file, sr=None)

    analysis_type = st.radio("Wybierz typ analizy:", ('Ramka', 'Klip'))

    if analysis_type == 'Klip':
        # Długość klipu
        clip_duration = st.selectbox("Wybierz długość klipu:", ["1s", "2s", "3s", "4s"])
        clip_duration = int(clip_duration[:-1])  # Usuń 's' i przekształć na int
        frame_size = clip_duration * sr  # Długość ramki w próbkach (np. 1s, 2s, 3s, 4s)
        hop_length = frame_size  # Przesunięcie ramki jest równe długości klipu
    
    elif analysis_type == 'Ramka':
        # Długość ramki
        frame_duration = st.selectbox("Wybierz długość ramki:", ["10ms", "20ms", "30ms", "40ms"])
        frame_duration = int(frame_duration[:-2])  # Usuń 'ms' i przekształć na int
        frame_size = int(frame_duration / 1000 * sr)  # Długość ramki w próbkach
        hop_length = frame_size  # Przesunięcie ramki jest równe długości ramki

    # Dodanie opcji detekcji ciszy
    silence_detection = st.selectbox("Wybierz opcję detekcji ciszy:", ["Brak", "Detekcja ciszy"])

    # Przygotowanie układu strony (szerokość)


    # Tworzenie dwóch kolumn (lewa strona - interfejs, prawa strona - wykresy)
    col1, col2 = st.columns([1, 3])  # Kolumna po lewej ma szerokość 1, po prawej 3

    with col1:
        # Przyciski i opcje po lewej stronie
        st.write("### Opcje analizy:")
        # Wybór analizy: klip czy ramka?
        analysis_type = st.radio("Wybierz typ analizy:", ('Klip', 'Ramka'), key="analysis_type_key")

        # Dodanie opcji detekcji ciszy
        silence_detection = st.selectbox("Wybierz opcję detekcji ciszy:", ["Brak", "Detekcja ciszy"], key="silence_detection_key")

        # Wybór długości klipu lub ramki (jeśli dotyczy)
        if analysis_type == 'Klip':
            clip_duration = st.selectbox("Wybierz długość klipu:", ["1s", "2s", "3s", "4s"], key="clip_duration_2")
            clip_duration = int(clip_duration[:-1])  # Usuń 's' i przekształć na int
        elif analysis_type == 'Ramka':
            frame_duration = st.selectbox("Wybierz długość ramki:", ["10ms", "20ms", "30ms", "40ms"], key="frame_duration_2")
            frame_duration = int(frame_duration[:-2])  # Usuń 'ms' i przekształć na int

    with col2:
        # Dodanie wykresu przebiegu czasowego audio
        st.write("### Wykres przebiegu czasowego audio:")
        time_axis = np.linspace(0, len(y) / sr, len(y))
        fig = go.Figure()

        # Dodanie podstawowego wykresu przebiegu czasowego
        fig.add_trace(go.Scatter(x=time_axis, y=y, mode='lines', name="Przebieg czasowy"))

        if silence_detection == "Detekcja ciszy":
            # Wykrywanie ciszy
            silence_flags = detect_silence(y, frame_size=frame_size, hop_length=hop_length)

            # Zaznaczenie ciszy na wykresie
            silence_start_times = [i * frame_size / sr for i, silent in enumerate(silence_flags) if silent]
            silence_end_times = [(i + 1) * frame_size / sr for i, silent in enumerate(silence_flags) if silent]
            y_min = min(y)
            y_max = max(y)
            
            for start, end in zip(silence_start_times, silence_end_times):
                # Zaznaczenie prostokątem wykrytych fragmentów ciszy
                fig.add_trace(go.Scatter(
                    x=[start, start, end, end], 
                    y=[y_min, y_max, y_max, y_min],  # Ustalamy wartości Y tak, żeby prostokąt obejmował całą wysokość wykresu
                    fill='toself',  # Wypełnienie prostokąta
                    fillcolor='rgba(255, 0, 0, 0.3)',  # Kolor wypełnienia (czerwony, przezroczysty)
                    line=dict(width=0),  # Brak konturu prostokąta
                    showlegend=False,  # Usunięcie legendy
                    mode='none',  # Usunięcie kropek i linii
                    name="Cisza"
                ))

        fig.update_layout(
            title="Przebieg czasowy sygnału audio" if silence_detection == "Brak" else "Przebieg czasowy z detekcją ciszy",
            xaxis_title="Czas (s)",
            yaxis_title="Amplituda",
            hovermode="closest"
        )
        st.plotly_chart(fig, use_container_width=True)


        # Wybór analizy: klip czy ramka?
        if analysis_type == 'Klip':

            # Obliczenia cech na poziomie klipu
            st.write("### Analiza na poziomie klipu:")
            frame_size = clip_duration * sr  # Długość ramki w próbkach (np. 1s, 2s, 3s, 4s)
            hop_length = frame_size  # Przesunięcie ramki jest równe długości klipu

            # Obliczanie cech
            vstd = compute_vstd(y, frame_size, hop_length)
            vdr = compute_vdr(y, frame_size, hop_length)
            lster = compute_lster(y, frame_size, hop_length)
            entropy = compute_energy_entropy(y, frame_size, hop_length)
            zstd = compute_zstd(y, frame_size, hop_length)
            hzcrr = compute_hzcrr(y, frame_size, hop_length)

            # Wykresy dla cech na poziomie klipu
            col1, col2, col3, col4, col5= st.columns(5)  # Sześć wykresów w poziomie

            with col1:
                st.plotly_chart(plot_feature(y, sr, vstd, "VSTD", frame_size, hop_length), use_container_width=True)

            with col2:
                st.plotly_chart(plot_feature(y, sr, vdr, "VDR", frame_size, hop_length), use_container_width=True)

            with col3:
                st.plotly_chart(plot_feature(y, sr, lster, "LSTER", frame_size, hop_length), use_container_width=True)

            with col4:
                st.plotly_chart(plot_feature(y, sr, entropy, "Entropy", frame_size, hop_length), use_container_width=True)

            with col5:
                st.plotly_chart(plot_feature(y, sr, zstd, "ZSTD", frame_size, hop_length), use_container_width=True)

        elif analysis_type == 'Ramka':
            # Obliczenie cech na poziomie ramki
            st.write("### Analiza na poziomie ramki:")
            volumes = compute_volume(y, frame_size, hop_length)
            ste = compute_ste(y, frame_size, hop_length)
            zcr = compute_zcr(y, frame_size, hop_length)
            f0 = find_f0(y, frame_size, hop_length)

            # Wykresy dla cech na poziomie ramki
            col1, col2, col3, col4 = st.columns(4)  # Cztery wykresy w poziomie

            with col1:
                st.plotly_chart(plot_feature(y, sr, volumes, "Volume", frame_size, hop_length), use_container_width=True)

            with col2:
                st.plotly_chart(plot_feature(y, sr, ste, "Short Time Energy (STE)", frame_size, hop_length), use_container_width=True)

            with col3:
                st.plotly_chart(plot_feature(y, sr, zcr, "Zero Crossing Rate (ZCR)", frame_size, hop_length), use_container_width=True)

            with col4:
                st.plotly_chart(plot_feature(y, sr, f0, "Fundamental Frequency (F0)", frame_size, hop_length), use_container_width=True)


# import streamlit as st
# import librosa
# import numpy as np
# import plotly.graph_objects as go
# from audio_analysis import (compute_volume, compute_ste, compute_zcr, 
#                             compute_vstd, compute_vdr, compute_lster, 
#                             plot_feature, detect_silence)

# # Tytuł aplikacji
# st.title("🎵 Analiza sygnału audio")

# # Wczytanie pliku audio
# uploaded_file = st.file_uploader("Wybierz plik WAV", type=["wav"])

# if uploaded_file is not None:
#     # Wczytanie pliku audio
#     y, sr = librosa.load(uploaded_file, sr=None)

#     analysis_type = st.radio("Wybierz typ analizy:", ('Ramka', 'Klip'))

#     if analysis_type == 'Klip':
#         # Długość klipu
#         clip_duration = st.selectbox("Wybierz długość klipu:", ["1s", "2s", "3s", "4s"])
#         clip_duration = int(clip_duration[:-1])  # Usuń 's' i przekształć na int
#         frame_size = clip_duration * sr  # Długość ramki w próbkach (np. 1s, 2s, 3s, 4s)
#         hop_length = frame_size  # Przesunięcie ramki jest równe długości klipu
    
#     elif analysis_type == 'Ramka':
#         # Długość ramki
#         frame_duration = st.selectbox("Wybierz długość ramki:", ["10ms", "20ms", "30ms", "40ms"])
#         frame_duration = int(frame_duration[:-2])  # Usuń 'ms' i przekształć na int
#         frame_size = int(frame_duration / 1000 * sr)  # Długość ramki w próbkach
#         hop_length = frame_size  # Przesunięcie ramki jest równe długości ramki

#     # Dodanie opcji detekcji ciszy
#     silence_detection = st.selectbox("Wybierz opcję detekcji ciszy:", ["Brak", "Detekcja ciszy"])

#     # Dodanie wykresu przebiegu czasowego audio
#     st.write("### Wykres przebiegu czasowego audio:")
#     time_axis = np.linspace(0, len(y) / sr, len(y))
#     fig = go.Figure()

#     # Dodanie podstawowego wykresu przebiegu czasowego
#     fig.add_trace(go.Scatter(x=time_axis, y=y, mode='lines', name="Przebieg czasowy"))

#     if silence_detection == "Detekcja ciszy":
#         # Wykrywanie ciszy
#         silence_flags = detect_silence(y, frame_size=frame_size, hop_length=hop_length)

#         # Zaznaczenie ciszy na wykresie
#         silence_start_times = [i * frame_size / sr for i, silent in enumerate(silence_flags) if silent]
#         silence_end_times = [(i + 1) * frame_size / sr for i, silent in enumerate(silence_flags) if silent]
#         y_min = min(y)
#         y_max = max(y)
        
#         for start, end in zip(silence_start_times, silence_end_times):
#     # Zaznaczenie prostokątem wykrytych fragmentów ciszy
#             fig.add_trace(go.Scatter(
#                 x=[start, start, end, end], 
#                 y=[y_min, y_max, y_max, y_min],  # Ustalamy wartości Y tak, żeby prostokąt obejmował całą wysokość wykresu
#                 fill='toself',  # Wypełnienie prostokąta
#                 fillcolor='rgba(255, 0, 0, 0.3)',  # Kolor wypełnienia (czerwony, przezroczysty)
#                 line=dict(width=0),  # Brak konturu prostokąta
#                 showlegend=False,  # Usunięcie legendy
#                 mode='none',  # Usunięcie kropek i linii
#                 name="Cisza"
#             ))


#     fig.update_layout(
#         title="Przebieg czasowy sygnału audio" if silence_detection == "Brak" else "Przebieg czasowy z detekcją ciszy",
#         xaxis_title="Czas (s)",
#         yaxis_title="Amplituda",
#         hovermode="closest"
#     )
#     st.plotly_chart(fig, use_container_width=True)

#     # Dodanie opcji odtwarzania pliku audio
#     st.audio(uploaded_file)

#     # Wybór analizy: klip czy ramka?
#     analysis_type = st.radio("Wybierz typ analizy:", ('Klip', 'Ramka'))

#     if analysis_type == 'Klip':

#         # Obliczenie cech na poziomie klipu
#         st.write("### Analiza na poziomie klipu:")
#         frame_size = clip_duration * sr  # Długość ramki w próbkach (np. 1s, 2s, 3s, 4s)
#         hop_length = frame_size  # Przesunięcie ramki jest równe długości klipu

#         volumes = compute_volume(y, frame_size, hop_length)
#         ste = compute_ste(y, frame_size, hop_length)
#         zcr = compute_zcr(y, frame_size, hop_length)

#         # Wykresy dla głośności, STE, ZCR na poziomie klipu
#         st.plotly_chart(plot_feature(y, sr, volumes, "Volume", frame_size, hop_length), use_container_width=True)
#         st.plotly_chart(plot_feature(y, sr, ste, "Short Time Energy (STE)", frame_size, hop_length), use_container_width=True)
#         st.plotly_chart(plot_feature(y, sr, zcr, "Zero Crossing Rate (ZCR)", frame_size, hop_length), use_container_width=True)

#         # Obliczenia cech na poziomie klipu
#         vstd = compute_vstd(volumes)
#         vdr = compute_vdr(volumes)
#         lster = compute_lster(ste, frame_size, hop_length)

#         # Wyświetlanie wyników dla klipu
#         st.write(f"VSTD (Volume Standard Deviation): {vstd:.4f}")
#         st.write(f"VDR (Volume Dynamic Range): {vdr:.4f}")
#         st.write(f"LSTER (Low Short Time Energy Ratio): {lster:.4f}")

#     elif analysis_type == 'Ramka':
#         frame_size = int(frame_duration / 1000 * sr)  # Długość ramki w próbkach
#         hop_length = frame_size  # Przesunięcie ramki jest równe długości ramki

#         # Obliczenie cech na poziomie ramki
#         st.write("### Analiza na poziomie ramki:")
#         volumes = compute_volume(y, frame_size, hop_length)
#         ste = compute_ste(y, frame_size, hop_length)
#         zcr = compute_zcr(y, frame_size, hop_length)

#         # Wykresy dla głośności, STE, ZCR na poziomie ramki
#         st.plotly_chart(plot_feature(y, sr, volumes, "Volume", frame_size, hop_length), use_container_width=True)
#         st.plotly_chart(plot_feature(y, sr, ste, "Short Time Energy (STE)", frame_size, hop_length), use_container_width=True)
#         st.plotly_chart(plot_feature(y, sr, zcr, "Zero Crossing Rate (ZCR)", frame_size, hop_length), use_container_width=True)
