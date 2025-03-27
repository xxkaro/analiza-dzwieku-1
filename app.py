import time
import streamlit as st
import librosa
import io
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from audio_analysis import extract_audio_features, plot_feature, detect_silence, classify_voicing, detect_music

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

st.title("Analiza sygnału audio")

uploaded_file = st.file_uploader("Wybierz plik WAV", type=["wav"])

clip_duration = None
frame_size = None
hop_length = None  

if uploaded_file is not None:
    st.audio(uploaded_file)
    y, sr = librosa.load(uploaded_file, sr=None)

    col1, col2 = st.columns([1, 3]) 

    with col1:
        st.write("### Opcje analizy:")
        analysis_type = st.radio("Wybierz typ analizy:", ('Ramka', 'Klip'), key="analysis_type_key")
        f0_method = "amdf"  

        if analysis_type == "Ramka":
            detection = st.selectbox("Wybierz opcję detekcji:", ["Brak", "Detekcja ciszy", "Dźwięczne / Bezdźwięczne"], key="detection_key")
            f0_method = st.radio("Wybierz metodę wyznaczania f0:", ["AMDF", "Autokorelacja"], key="f0_method_key")
            if f0_method == "AMDF":
                f0_method = "amdf"
            elif f0_method == "Autokorelacja":
                f0_method = "autocorrelation"

        elif analysis_type == "Klip":
            detection = st.selectbox("Wybierz opcję detekcji:", ["Brak", "Muzyka / Mowa"], key="detection_key")
        
        if detection == "Detekcja ciszy":
            silence_zcr_threshold = st.slider("Wybierz próg dla ZCR:", min_value=0.0, max_value=0.1, step=0.0005, value=0.01, format="%0.3f")
            silence_volume_threshold = st.slider("Wybierz próg dla Volume:", min_value=0.0, max_value=0.1, step=0.0005, value=0.003, format="%0.3f")

        if detection == "Dźwięczne / Bezdźwięczne":
            voicing_ste_threshold = st.slider("Wybierz próg dla STE:", min_value=0.0, max_value=0.05, step=0.0005, value=0.0005, format="%0.3f")
            voicing_volume_threshold = st.slider("Wybierz próg dla Volume:", min_value=0.0, max_value=0.05, step=0.0005, value=0.005, format="%0.3f")

        if detection == "Muzyka / Mowa":    
            music_lster_threshold = st.slider("Wybierz próg dla LSTER:", min_value=0.0, max_value=1.0, step=0.005, value=0.5, format="%0.3f")
            music_zstd_threshold = st.slider("Wybierz próg dla ZSTD:", min_value=0.0, max_value=0.5, step=0.005, value=0.15, format="%0.3f")


        frame_duration = st.selectbox("Wybierz długość ramki:", ["10ms", "20ms", "30ms", "40ms"], key="frame_duration_2")
        frame_duration = int(frame_duration[:-2])  

        if analysis_type == 'Klip':
            clip_duration = 1000
            frame_size = clip_duration / 1000 * sr 
            hop_length = frame_size 
        elif analysis_type == 'Ramka':
            frame_size = int(frame_duration / 1000 * sr)  
            hop_length = frame_size  
            
        frame_features, clip_features = extract_audio_features(y, sr, hop_length, frame_size_ms=frame_duration, method=f0_method) 

    with col2:
        st.write("### Wykres przebiegu czasowego audio:")
        time_axis = np.linspace(0, len(y) / sr, len(y))
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=time_axis,
                                y=y,
                                mode='lines', 
                                name="Przebieg czasowy", 
                                hovertemplate="<b>Czas:</b> %{x:.3f} s" + "<br>" +
                                            "<b>Amplituda:</b> %{y:.3f}" + "<extra></extra>", 
        ))
        fig.update_layout(
            title="Przebieg czasowy",
            title_x=0.5,
            xaxis_title="Czas (s)",
            yaxis_title="Amplituda",
            hovermode="closest"  
        )

        if detection == "Detekcja ciszy":

            frame_features, clip_features = extract_audio_features(y, sr, hop_length, frame_size_ms=frame_duration, silence_zcr_threshold=silence_zcr_threshold, silence_volume_threshold=silence_volume_threshold)

            silence_flags = frame_features['is_silence']  

            silence_start_times = [min(i * frame_size / sr, len(y) / sr) for i, silent in enumerate(silence_flags) if silent]
            silence_end_times = [min((i+1) * frame_size / sr, len(y) / sr) for i, silent in enumerate(silence_flags) if silent]
            
            y_min = min(y)
            y_max = max(y)

            for i, (start, end) in enumerate(zip(silence_start_times, silence_end_times)):
                fig.add_trace(go.Scatter(
                    x=[start, start, end, end],
                    y=[y_min, y_max, y_max, y_min],
                    fill='toself', 
                    fillcolor='rgba(255, 0, 0, 0.3)',  
                    line=dict(width=0),  
                    showlegend=(i == 0),  
                    legendgroup="Cisza", 
                    mode='none', 
                    name="Cisza", 
                    hoverinfo="skip"
                ))

        if detection == "Dźwięczne / Bezdźwięczne":
            frame_features, clip_features = extract_audio_features(y, sr, hop_length, frame_size_ms=frame_duration, voicing_ste_threshold=voicing_ste_threshold, voicing_volume_threshold=voicing_volume_threshold)

            voicing_class = frame_features['is_voiced']

            start_times = [min(i * frame_size / sr, len(y) / sr) for i, voiced in enumerate(voicing_class)]
            end_times = [min((i+1) * frame_size / sr, len(y) / sr) for i, voiced in enumerate(voicing_class)]
            
            y_min = min(y)
            y_max = max(y)

            dźwięczne_legend_added = False
            bezdźwięczne_legend_added = False

            for i, (start, end, is_voiced) in enumerate(zip(start_times, end_times, voicing_class)):
                if is_voiced:
                    fillcolor = 'rgba(0, 255, 0, 0.3)'
                    label = "Dźwięczne"
                    legendgroup = "Dźwięczne"

                    show_legend = not dźwięczne_legend_added
                    if not dźwięczne_legend_added:
                        dźwięczne_legend_added = True

                else:
                    fillcolor = 'rgba(255, 0, 0, 0.3)' 
                    label = "Bezdźwięczne"
                    legendgroup = "Bezdźwięczne"  

                    show_legend = not bezdźwięczne_legend_added
                    if not bezdźwięczne_legend_added:
                        bezdźwięczne_legend_added = True

                fig.add_trace(go.Scatter(
                    x=[start, start, end, end], 
                    y=[y_min, y_max, y_max, y_min],  
                    fill='toself', 
                    fillcolor=fillcolor, 
                    line=dict(width=0),
                    showlegend=show_legend, 
                    legendgroup=legendgroup, 
                    mode='none', 
                    name=label  
                ))


        if detection == "Muzyka / Mowa":
            frame_features, clip_features = extract_audio_features(y, sr, hop_length, frame_size_ms=frame_duration, music_lster_threshold=music_lster_threshold, music_zstd_threshold=music_zstd_threshold)

            music_flags = clip_features['is_music']

            start_times = [min(i * frame_size / sr, len(y) / sr) for i, music in enumerate(music_flags)]
            end_times = [min((i + 1) * frame_size / sr, len(y) / sr) for i, music in enumerate(music_flags)]
            
            y_min = min(y)
            y_max = max(y)

            muzyka_legend_added = False
            mowa_legend_added = False

            for i, (start, end, is_music) in enumerate(zip(start_times, end_times, music_flags)):
                if is_music:
                    fillcolor = 'rgba(0, 255, 0, 0.3)' 
                    label = "Muzyka"
                    legendgroup = "Muzyka"  

                    show_legend = not muzyka_legend_added
                    if not muzyka_legend_added:
                        muzyka_legend_added = True

                else:
                    fillcolor = 'rgba(255, 0, 0, 0.3)'
                    label = "Mowa"
                    legendgroup = "Mowa"  

                    show_legend = not mowa_legend_added
                    if not mowa_legend_added:
                        mowa_legend_added = True

                fig.add_trace(go.Scatter(
                    x=[start, start, end, end], 
                    y=[y_min, y_max, y_max, y_min], 
                    fill='toself', 
                    fillcolor=fillcolor, 
                    line=dict(width=0), 
                    showlegend=show_legend,
                    legendgroup=legendgroup,
                    mode='none', 
                    name=label, 
                    hoverinfo="skip"
                ))


        fig.update_layout(
            title="Przebieg czasowy sygnału audio",
            xaxis_title="Czas (s)",
            yaxis_title="Amplituda",
            hovermode="closest",
        )

        st.plotly_chart(fig, use_container_width=True)


    if analysis_type == 'Klip':
        st.write("### Analiza na poziomie klipu:")
        frame_size = clip_duration * sr 
        hop_length = frame_size  

        features = [
            ('VSTD', clip_features['vstd']),
            ('VDR', clip_features['vdr']),
            ('LSTER', clip_features['lster']),
            ('Entropy', clip_features['energy_entropy']),
            ('ZSTD', clip_features['zstd'])
        ]

        num_features = len(features)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_feature(y, sr, features[0][1], features[0][0], "klip"), use_container_width=True)
        with col2:
            st.plotly_chart(plot_feature(y, sr, features[1][1], features[1][0], "klip"), use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(plot_feature(y, sr, features[2][1], features[2][0], "klip"), use_container_width=True)
        with col4:
            st.plotly_chart(plot_feature(y, sr, features[3][1], features[3][0], "klip"), use_container_width=True)

        if num_features % 2 != 0:
            col5, col6, col7 = st.columns([1, 2, 1]) 
            with col6:  
                st.plotly_chart(plot_feature(y, sr, features[4][1], features[4][0], "klip"), use_container_width=True)



    elif analysis_type == 'Ramka':
        st.write("### Analiza na poziomie ramki:")

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_feature(y, sr, frame_features['volume'], "Volume", "ramka"), use_container_width=True)
        with col2:
            st.plotly_chart(plot_feature(y, sr, frame_features['ste'], "STE", "ramka"), use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(plot_feature(y, sr, frame_features['zcr'], "ZCR", "ramka"), use_container_width=True)
        with col4:
            st.plotly_chart(plot_feature(y, sr, frame_features['f0'], "f0", "ramka"), use_container_width=True)


    if analysis_type == 'Klip':
        df = pd.DataFrame(clip_features)
    else: 
        df = pd.DataFrame(frame_features)

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()

    st.write("### Podgląd danych CSV:")
    st.dataframe(df.head(5))

    st.download_button(
        label="📥 Pobierz CSV",
        data=csv_data,
        file_name="audio_analysis.csv",
        mime="text/csv"
)
