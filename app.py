import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# Función TOPSIS
# ==============================
def topsis(df, pesos, beneficio, costo):
    matriz = df.iloc[:, 1:].values
    norm = np.sqrt((matriz**2).sum(axis=0))
    matriz_normalizada = matriz / norm
    matriz_ponderada = matriz_normalizada * pesos

    ideal_pos, ideal_neg = [], []
    criterios = df.columns[1:]

    for i, crit in enumerate(criterios):
        if crit in beneficio:
            ideal_pos.append(matriz_ponderada[:, i].max())
            ideal_neg.append(matriz_ponderada[:, i].min())
        else:
            ideal_pos.append(matriz_ponderada[:, i].min())
            ideal_neg.append(matriz_ponderada[:, i].max())

    ideal_pos, ideal_neg = np.array(ideal_pos), np.array(ideal_neg)
    dist_pos = np.sqrt(((matriz_ponderada - ideal_pos)**2).sum(axis=1))
    dist_neg = np.sqrt(((matriz_ponderada - ideal_neg)**2).sum(axis=1))
    score = dist_neg / (dist_pos + dist_neg)

    df["Score TOPSIS"] = score
    df["Ranking"] = df["Score TOPSIS"].rank(ascending=False)
    return df

# ==============================
# Función AHP para calcular pesos
# ==============================
def ahp_weights(matrix):
    col_sum = matrix.sum(axis=0)
    norm_matrix = matrix / col_sum
    weights = norm_matrix.mean(axis=1)
    return weights / weights.sum()

# ==============================
# Interfaz Streamlit
# ==============================
st.title("DecisionLab")

# Párrafo introductorio
st.markdown(
    """
    **DecisionLab** es un sistema de apoyo a la toma de decisiones desarrollado en Python 
    que integra los métodos **AHP** y **TOPSIS** para evaluar y jerarquizar alternativas 
    de manera objetiva, considerando múltiples criterios de decisión.
    """
)

# Subir CSV
archivo = st.file_uploader("📂 Puedes subir un archivo CSV, o editar la tabla.", type=["csv"])

if archivo:
    st.session_state["tabla"] = pd.read_csv(archivo)
    st.success("Archivo cargado correctamente ✅")

# Estado persistente de la tabla si no hay CSV
if "tabla" not in st.session_state:
    st.session_state["tabla"] = pd.DataFrame({
        "Elementos": ["Opción1", "Opción2"],
        "Criterio1": [0.0, 0.0],
        "Criterio2": [0.0, 0.0]
    })

# Botones para agregar/eliminar filas y columnas
st.markdown("### 1. Edita tu tabla de criterios y opciones")
col1, col2, col3, col4 = st.columns(4)

if col1.button("➕ Agregar fila"):
    nueva_fila = {c: 0.0 for c in st.session_state["tabla"].columns}
    nueva_fila["Elementos"] = f"Opción{len(st.session_state['tabla'])+1}"
    st.session_state["tabla"] = pd.concat(
        [st.session_state["tabla"], pd.DataFrame([nueva_fila])],
        ignore_index=True
    )

# Input para nombre de nueva columna
nuevo_nombre = col2.text_input("Nombre del nuevo criterio", "")
if col2.button("➕ Agregar columna"):
    if nuevo_nombre.strip() == "":
        st.warning("Por favor escribe un nombre para el criterio antes de agregarlo.")
    elif nuevo_nombre in st.session_state["tabla"].columns:
        st.error("Ese nombre de criterio ya existe, elige otro.")
    else:
        st.session_state["tabla"][nuevo_nombre] = 0.0

# Eliminar columna
columnas = [c for c in st.session_state["tabla"].columns if c != "Elementos"]
columna_a_eliminar = col3.selectbox("Columna a eliminar", [""] + columnas)
if col3.button("🗑️ Eliminar columna") and columna_a_eliminar:
    st.session_state["tabla"].drop(columns=[columna_a_eliminar], inplace=True)

# Eliminar fila
fila_a_eliminar = col4.selectbox("Fila a eliminar", [""] + list(st.session_state["tabla"]["Elementos"]))
if col4.button("🗑️ Eliminar fila") and fila_a_eliminar:
    st.session_state["tabla"] = st.session_state["tabla"][st.session_state["tabla"]["Elementos"] != fila_a_eliminar]

# Editor de tabla
data = st.data_editor(st.session_state["tabla"])
st.session_state["tabla"] = data

# ==============================
# Comparaciones AHP con slider etiquetado
# ==============================
st.markdown("### 2. Comparaciones AHP entre criterios")
criterios = list(data.columns[1:])
n = len(criterios)

matrix = np.ones((n, n))

for i in range(n):
    for j in range(i+1, n):
        opcion = st.radio(
            f"Entre {criterios[i]} y {criterios[j]}, ¿cuál es más importante?",
            options=[criterios[i], criterios[j]],
            key=f"radio-{i}-{j}"
        )

        # Slider con etiquetas en cada punto
        nivel = st.select_slider(
            f"Nivel de importancia de {opcion} respecto al otro",
            options=["Moderada (3)", "Fuerte (5)", "Muy Fuerte (7)", "Extremo (9)"],
            value="Moderada (3)",
            key=f"slider-{i}-{j}"
        )

        # Mapear selección a valor numérico
        mapa_valores = {
            "Moderada (3)": 3,
            "Fuerte (5)": 5,
            "Muy Fuerte (7)": 7,
            "Extremo (9)": 9
        }
        val = mapa_valores[nivel]

        # Llenar matriz según selección
        if opcion == criterios[i]:
            matrix[i, j] = val
            matrix[j, i] = 1/val
        else:
            matrix[j, i] = val
            matrix[i, j] = 1/val

pesos = ahp_weights(matrix)
st.write("### Pesos calculados con AHP")
st.dataframe(pd.DataFrame({"Criterio": criterios, "Peso": pesos}))

# ==============================
# Beneficio / Costo
# ==============================
st.markdown("### 3. Define qué valores se quieren maximizar, los demás se minimizarán.")
beneficio = st.multiselect("Criterios a Maximizar", criterios, default=criterios)
costo = [c for c in criterios if c not in beneficio]

# ==============================
# Calcular TOPSIS
# ==============================
st.markdown("### 4. Ejecutar TOPSIS")
if st.button("Calcular"):
    try:
        resultado = topsis(data.copy(), pesos, beneficio, costo)
        st.write("### Resultados")
        st.dataframe(resultado[["Elementos","Score TOPSIS","Ranking"]])

        # Gráfica
        fig, ax = plt.subplots()
        ax.bar(resultado["Elementos"], resultado["Score TOPSIS"])
        ax.set_title("Ranking TOPSIS")
        ax.set_ylabel("Score")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")  # línea divisoria opcional
st.markdown(
    "<p style='text-align: center; font-size: 12px; color: gray;'>Desarrollado por Micaela Corrales</p>",
    unsafe_allow_html=True
)