import numpy as np

def extract_ppg_features(raw_intensities, fps=30):
    """
    Processa a lista de intensidades de brilho (canal vermelho) do vídeo
    e extrai features cruciais para o modelo de estimativa de pressão arterial.
    
    :param raw_intensities: list[float] - Valores de brilho médio extraídos quadro a quadro
    :param fps: int - Taxa de quadros por segundo da câmera (padrão: 30)
    :return: dict - Dicionário contendo amplitude, tempo de subida e frequência cardíaca estimada
    """
    if not raw_intensities or len(raw_intensities) < 60:
        # Fallback padrão caso a captura seja muito curta
        return {
            "mean_amplitude": 0.0,
            "mean_intensity": 0.0,
            "rise_time": 0.0,
            "heart_rate": 0
        }

    signal = np.array(raw_intensities, dtype=float)

    # 1. Suavização simples do sinal para remover ruído de tremulação (Média Móvel)
    window_size = 3
    smoothed = np.convolve(signal, np.ones(window_size)/window_size, mode='valid')

    # 2. Métricas de Amplitude e Intensidade
    max_val = float(np.max(smoothed))
    min_val = float(np.min(smoothed))
    mean_val = float(np.mean(smoothed))
    amplitude = max_val - min_val

    # 3. Detecção de picos simples (Sístoles)
    # Identifica pontos locais que são maiores que seus vizinhos e acima da média
    peaks = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1] and smoothed[i] > mean_val:
            peaks.append(i)

    # 4. Cálculo da Frequência Cardíaca (BPM) e Tempo de Subida da Onda
    heart_rate = 70  # Valor padrão de fallback
    rise_time = 0.15 # Tempo de subida médio estimado (em segundos)

    if len(peaks) >= 2:
        # Intervalos entre picos (inter-beat intervals)
        intervals = np.diff(peaks) / fps
        avg_interval = np.mean(intervals)
        if avg_interval > 0:
            heart_rate = int(round(60.0 / avg_interval))
            
            # Limita os valores de BPM dentro de faixas fisiológicas normais
            heart_rate = max(40, min(180, heart_rate))
            
            # Estimativa do tempo de subida (aproximadamente 20% a 30% do ciclo cardíaco)
            rise_time = round(float(avg_interval * 0.25), 3)

    return {
        "mean_amplitude": round(float(amplitude), 4),
        "mean_intensity": round(float(mean_val), 4),
        "rise_time": rise_time,
        "heart_rate": heart_rate
    }