import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

st.set_page_config(layout="wide")
st.title("🦺 Detección de Equipos de Seguridad con YOLOv8")

# Cargar modelos
person_model = YOLO("yolov8n.pt")
ppe_model = YOLO("best.pt")

# Entrada de imagen
option = st.radio("Selecciona entrada:", ["📷 Cámara", "🖼️ Imagen"])
if option == "📷 Cámara":
    img_file = st.camera_input("Captura una imagen")
else:
    img_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

# Procesamiento
if img_file:
    image = Image.open(img_file).convert("RGB")
    image_np = np.array(image)

    st.subheader("📍 Paso 1: Detección de Personas")
    results_person = person_model.predict(image_np, classes=[0], conf=0.4)
    boxes = results_person[0].boxes.xyxy.cpu().numpy().astype(int)

    if len(boxes) == 0:
        st.warning("No se detectaron personas.")
    else:
        st.success(f"{len(boxes)} persona(s) detectada(s).")

        # Mostrar primero imagen general con todas las personas
        st.subheader("👥 Vista general de personas detectadas")
        img_with_boxes = results_person[0].plot()
        st.image(img_with_boxes, caption="Personas detectadas", use_column_width=True)

        # Para cada persona detectada
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            st.markdown(f"---\n### 🧍 Persona #{i+1}")

            person_crop = image_np[y1:y2, x1:x2]
            results_ppe = ppe_model.predict(person_crop, conf=0.4)
            img_annotated = results_ppe[0].plot()

            # Mostrar imagen de persona con elementos detectados (tamaño más pequeño)
            st.image(img_annotated, caption="Elementos de seguridad detectados", width=300)

            # Obtener clases detectadas
            class_ids = results_ppe[0].boxes.cls.cpu().numpy().astype(int)
            class_names = [ppe_model.names[idx] for idx in class_ids]

            # Lista de EPP esperados (puedes modificar según tu modelo)
            epp_esperado = {"botas", "casco", "chaleco","trabajador","guantes"}  # Ejemplo
            epp_detectado = set(class_names)
            epp_faltantes = epp_esperado - epp_detectado

            st.markdown(f"**✅ Detectado:** {', '.join(epp_detectado) if epp_detectado else 'Nada'}")
            st.markdown(f"**❌ Faltantes:** {', '.join(epp_faltantes) if epp_faltantes else 'Ninguno'}")
