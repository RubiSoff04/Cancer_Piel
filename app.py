import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import random
from datetime import datetime, timedelta
from PIL import Image
import io

# Cargar el modelo
@st.cache_resource
def cargar_modelo():
    return load_model("modelo_skin_cancer.h5")

modelo = cargar_modelo()

# Función para predecir usando el modelo real
def detectar_cancer_real(img_data):
    img = Image.open(img_data).convert('RGB')
    img = img.resize((224, 224))  # Tamaño esperado del modelo
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediccion = modelo.predict(img_array)[0][0]
    return "Maligno" if prediccion >= 0.5 else "Benigno"

# Generar cita aleatoria
def generar_cita():
    fecha = datetime.now() + timedelta(days=random.randint(1, 7))
    hora = random.choice(["08:00", "10:00", "13:00", "15:00"])
    return fecha.strftime("%d/%m/%Y"), hora

# Configuración general
st.set_page_config(page_title="Detección de Cáncer de Piel", layout="centered")
st.title("🧴 App de Detección de Cáncer de Piel")

# Estado de navegación entre pantallas
if "datos_registrados" not in st.session_state:
    st.session_state.datos_registrados = False

# FASE 1 - REGISTRO
if not st.session_state.datos_registrados:
    with st.form("form_datos"):
        st.subheader("🧾 Ingrese sus datos personales")

        nombre = st.text_input("Nombre")
        apellido = st.text_input("Apellido")
        dni = st.text_input("DNI", max_chars=8)
        fecha_nac = st.date_input("Fecha de nacimiento")

        departamento = st.selectbox("Departamento", ["Junín", "Lima", "Cusco"])
        provincia = st.text_input("Provincia")
        distrito = st.text_input("Distrito")

        enviado = st.form_submit_button("Siguiente")

        if enviado:
            if nombre and apellido and dni and len(dni) == 8:
                st.session_state.datos_registrados = True
                st.success("✅ Datos registrados correctamente. Ahora puede subir su imagen.")
            else:
                st.error("⚠️ Complete todos los campos correctamente.")

# FASE 2 - SUBIDA Y ANÁLISIS DE IMAGEN
if st.session_state.datos_registrados:
    st.subheader("📸 Suba una imagen de su piel")
    imagen = st.file_uploader("Seleccionar imagen", type=["jpg", "png", "jpeg"])

    if imagen is not None:
        st.image(imagen, caption="Imagen cargada", use_column_width=True)

        if st.button("🔍 Analizar imagen"):
            resultado = detectar_cancer_real(imagen)
            st.write(f"🧪 Resultado del análisis: **{resultado}**")

            if resultado == "Maligno":
                st.warning("⚠️ Se detectó una posible anomalía. Se sugiere una cita médica.")

                fecha_cita, hora_cita = generar_cita()
                st.markdown("### 🏥 Clínica: Medisalud Huancayo")
                st.markdown("**Especialidad:** Dermatología Oncológica")
                st.markdown("**Doctor:** Dr. Elías Mendoza")
                st.markdown(f"**Fecha:** {fecha_cita}")
                st.markdown(f"**Hora:** {hora_cita}")

                aceptar = st.button("✅ Aceptar cita")
                rechazar = st.button("❌ Rechazar cita")

                if aceptar:
                    st.success("Cita confirmada. ¡Gracias por confiar en nosotros!")
                elif rechazar:
                    st.info("Cita rechazada. Puede volver a intentarlo más adelante.")
            else:
                st.success("No se detectaron anomalías. Su piel parece saludable.")