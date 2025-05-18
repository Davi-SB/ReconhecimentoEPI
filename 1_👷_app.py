from PIL import Image
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from datetime import datetime

st.set_page_config(
    page_title="Reconhecimento Capacete",  # título da aba
    layout="wide",
    page_icon="⛑️",  # ícone da aba           
)

# Carrega modelo só uma vez
if 'model' not in st.session_state:
    st.session_state.model = load_model('keras_model.h5')
model = st.session_state.model
classes = ['Com Capacete', 'Sem Capacete', 'Capacete Falso']

st.title("⛑️ Registro de Entrada com Reconhecimento de Capacete")

# Inicializa histórico e contador de foto na sessão
if 'historico' not in st.session_state:
    st.session_state.historico = []
if 'foto_counter' not in st.session_state:
    st.session_state.foto_counter = 0

# Cria as colunas
col1, col2 = st.columns(2)

with col1:
    img_file = st.camera_input(
        "Tire uma foto para verificação de capacete",
        key=st.session_state.foto_counter
    )

with col2:
    # Se uma foto foi tirada
    if img_file is not None:
        img = Image.open(img_file)
        img = img.resize((224, 224))
        img_array = np.asarray(img)
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array

        # Prediz
        prediction = model.predict(data)
        indexVal = np.argmax(prediction)
        classe_detectada = classes[indexVal]

        if classe_detectada == 'Com Capacete':
            st.success("Capacete detectado! Bem-vindo(a).")
            login = st.text_input("Digite seu login para registrar a entrada:")
            if st.button("Registrar entrada"):
                if login.strip() != "":
                    horario = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                    entrada = f"{login} entrou com sucesso às {horario}."
                    st.session_state.historico.append(entrada)
                    st.success("Entrada registrada com sucesso!")
                    # Reseta câmera para próxima pessoa
                    st.session_state.foto_counter += 1
                    st.rerun()
                else:
                    st.warning("Por favor, preencha o login.")
        else:
            st.error(f"{classe_detectada} detectado. Por favor, coloque o capacete para prosseguir.")

    # Exibe histórico
    if st.session_state.historico:
        st.subheader("Histórico de entradas")
        for item in st.session_state.historico:
            st.write(item)
