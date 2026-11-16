# CardiaSync PWA

Sistema experimental de fotopletismografia (PPG) por câmera do celular, com:

- Preview ao vivo da câmera + guia visual do dedo
- Detecção básica de qualidade do sinal
- Contador + barra de progresso
- Fluxo de **Calibração** (medidor de pressão → câmera)
- Tabela de calibrações e placeholder de modelo de regressão
- Histórico de medições
- Pronto para deploy no **Render.com** + **Neon PostgreSQL**

> ⚠️ **Não é um dispositivo médico.** As medições são estimativas experimentais.

---

## Estrutura

```
cardiasync/
├── app.py                 # Flask + rotas + APIs
├── models.py              # SQLAlchemy (User, Calibration, Measurement, CalibrationModel)
├── calibration_engine.py  # Extração de features + placeholder do modelo
├── requirements.txt
├── .env.example
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── measure.html       # Tela de medição com preview
│   ├── calibrate.html     # Fluxo Medir aparelho → Calibrar câmera
│   └── historico.html
└── static/
    ├── css/style.css
    ├── js/measure.js
    ├── js/calibrate.js
    ├── icons/icon.svg     # Ícone original (coração + ECG)
    └── manifest.json
```

---

## Deploy no Render.com

1. Crie um repositório Git com esta pasta.
2. No Render: **New → Web Service** → conecte o repo.
3. Configurações:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT`
4. Environment Variables:
   - `DATABASE_URL` → connection string do Neon (postgresql://...)
   - `SECRET_KEY` → qualquer string forte
5. O Render já fornece **HTTPS**, então a câmera funciona no celular.

### Banco Neon

- Crie um projeto em [neon.tech](https://neon.tech)
- Copie a connection string e cole em `DATABASE_URL`
- As tabelas são criadas automaticamente na primeira requisição.

---

## Fluxo de uso

### Calibrar
1. Meça a pressão com o esfigmomanômetro.
2. Abra **Calibrar** → informe sistólica e diastólica.
3. Capture 15 s com o dedo na câmera + flash.
4. O par é salvo na tabela `calibrations`.

### Medir
1. Abra **Medir**.
2. Posicione o dedo (veja o preview e o círculo-guia).
3. Inicie a captura.
4. O sistema calcula FC e tenta estimar PA (se houver modelo calibrado).

---

## Modelo de calibração (placeholder)

Arquivo `calibration_engine.py`:

- `extract_basic_features()` → HR, qualidade, amplitude, etc.
- `estimate_blood_pressure()` → retorna `None` enquanto não houver coeficientes treinados.
- `train_simple_linear_model()` → placeholder para treinar quando ≥ 5 calibrações.

Quando quiser ativar a regressão de verdade, implemente o treino com `scikit-learn` e salve os coeficientes na tabela `calibration_models`.

---

## Ícone

Ícone SVG original: coração com linha de ECG/pulso em gradiente azul-ciano, sobre fundo escuro. Funciona como favicon e ícone de PWA.
