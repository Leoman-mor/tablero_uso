# app.py
# -----------------------------------------------------------
# Dashboard de Usabilidad Soluquim
#   - Lee empresas.csv (misma carpeta)
#   - Sesiones por usuario+empresa
#   - Modo de sesión: por Ingreso a login o por inactividad
#   - Métricas: frecuencia, profundidad, conversión, ranking
# -----------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# -----------------------------------------------------------
# CONFIG GENERAL
# -----------------------------------------------------------
st.set_page_config(
    page_title="Usabilidad Soluquim",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .big-metric {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: -8px;
    }
    .metric-label {
        color: #666;
        font-size: 13px;
    }
    .small-help {
        color: #888;
        font-size: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# CARGA DE DATOS
# -----------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("empresas.csv")
    # Columnas esperadas: function_used, created_at, usuario, empresa
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["fecha"] = df["created_at"].dt.date
    df = df.sort_values(["empresa", "usuario", "created_at"])
    return df

df_raw = load_data()

# -----------------------------------------------------------
# FUNCIONES AUXILIARES
# -----------------------------------------------------------
def preparar_sesiones(
    df: pd.DataFrame,
    session_timeout: int = 30,
    modo_sesion: str = "ingreso",
) -> pd.DataFrame:
    """
    Crea session_id por empresa+usuario sobre el DF YA FILTRADO.

    - modo_sesion = "ingreso":
        Cada evento 'Ingreso a login' inicia nueva sesión por empresa+usuario.
    - modo_sesion = "inactividad":
        Se corta sesión por tiempo de inactividad > session_timeout (min).
    """
    df = df.copy().sort_values(["empresa", "usuario", "created_at"])
    group_keys = ["empresa", "usuario"]

    if modo_sesion == "ingreso":
        df["es_ingreso"] = df["function_used"].str.contains(
            "Ingreso a login", case=False, na=False
        )
        df["sesion_local"] = df.groupby(group_keys)["es_ingreso"].cumsum()
        df["sesion_local"] = df["sesion_local"].where(df["sesion_local"] > 0, 1)
        df["diff_min"] = np.nan
        df["nueva_sesion"] = df["es_ingreso"]

    elif modo_sesion == "inactividad":
        df["diff_min"] = (
            df.groupby(group_keys)["created_at"]
            .diff()
            .dt.total_seconds()
            .div(60)
        )
        df["nueva_sesion"] = df["diff_min"].isna() | (df["diff_min"] > session_timeout)
        df["sesion_local"] = df.groupby(group_keys)["nueva_sesion"].cumsum()
    else:
        raise ValueError("modo_sesion debe ser 'ingreso' o 'inactividad'.")

    # session_id legible pero único
    df["session_id"] = (
        df["empresa"].astype(str)
        + " | U:"
        + df["usuario"].astype(str)
        + " | S:"
        + df["sesion_local"].astype(str)
    )

    # Flags de acciones clave
    df["es_ver"] = df["function_used"].eq("Visualización del detalle de un producto")
    df["es_fds"] = df["function_used"].eq("Descarga de FDS")
    df["es_etq"] = df["function_used"].eq("Descarga de etiquetas")

    return df


def bucket_acciones(n: int) -> str:
    """Agrupa número de acciones en rangos para profundidad de uso."""
    if n == 1:
        return "1 acción"
    elif n <= 3:
        return "2–3 acciones"
    elif n <= 5:
        return "4–5 acciones"
    elif n <= 10:
        return "6–10 acciones"
    else:
        return "11+ acciones"


# -----------------------------------------------------------
# TÍTULO Y DESCRIPCIÓN
# -----------------------------------------------------------
st.title("📊 Dashboard de Usabilidad – Soluquim")
st.caption(
    "Explora cómo las empresas y usuarios usan la plataforma: frecuencia, profundidad, conversión y ranking de usabilidad."
)

with st.expander("ℹ️ Glosario rápido"):
    st.markdown(
        """
- **Sesión:** grupo de acciones de un mismo usuario dentro de una empresa.
- **Profundidad:** cuántas acciones se hacen en promedio en cada sesión.
- **Conversión:** de las sesiones donde se ve un producto, ¿en qué porcentaje se hace una descarga (FDS o etiqueta)?
- **Índice de usabilidad:** mezcla de frecuencia (sesiones), profundidad y conversión para comparar empresas.
        """
    )

# -----------------------------------------------------------
# SIDEBAR – FILTROS LIMPIOS
# -----------------------------------------------------------
st.sidebar.header("🎯 Alcance del análisis")

# 1) Rango de fechas (primero)
min_fecha = df_raw["fecha"].min()
max_fecha = df_raw["fecha"].max()

rango_fechas = st.sidebar.date_input(
    "Periodo de análisis",
    value=(min_fecha, max_fecha),
    min_value=min_fecha,
    max_value=max_fecha,
    help="Solo se consideran eventos dentro de este rango de fechas.",
)

if isinstance(rango_fechas, tuple):
    fecha_ini, fecha_fin = rango_fechas
else:
    fecha_ini, fecha_fin = rango_fechas, rango_fechas

# 2) Empresa (una o todas, sin lista gigante)
empresas_unicas = sorted(df_raw["empresa"].unique())
opciones_emp = ["Todas las empresas"] + empresas_unicas
empresa_sel = st.sidebar.selectbox(
    "Empresa",
    options=opciones_emp,
    help="Puedes ver todas las empresas o enfocarte en una sola.",
)

if empresa_sel == "Todas las empresas":
    empresas_sel = empresas_unicas
else:
    empresas_sel = [empresa_sel]

st.sidebar.markdown("---")

# 3) Cómo se cuenta una sesión
st.sidebar.subheader("⚙️ Sesiones")

modo_sesion_label = st.sidebar.radio(
    "¿Cómo quieres contar una sesión?",
    options=[
        "Cada vez que el usuario inicia sesión (Ingreso a login)",
        "Por inactividad (si pasa cierto tiempo sin eventos)",
    ],
)

if modo_sesion_label.startswith("Cada vez"):
    modo_sesion = "ingreso"
    session_timeout = 30  # no se usa realmente en este modo
    st.sidebar.caption("Sesión = desde un 'Ingreso a login' hasta el siguiente.")
else:
    modo_sesion = "inactividad"
    session_timeout = st.sidebar.slider(
        "Corte por inactividad (minutos)",
        min_value=5,
        max_value=90,
        value=30,
        step=5,
        help="Si pasan más minutos sin eventos de ese usuario, se inicia una nueva sesión.",
    )
    st.sidebar.caption("Sesión = bloque continuo de actividad sin pausas largas.")

# -----------------------------------------------------------
# APLICAR FILTROS A LOS DATOS BASE
# -----------------------------------------------------------
df_filt = df_raw[
    (df_raw["empresa"].isin(empresas_sel))
    & (df_raw["fecha"].between(fecha_ini, fecha_fin))
].copy()

if df_filt.empty:
    st.warning("No hay datos para la empresa/periodo seleccionado.")
    st.stop()

# Construir sesiones SOBRE el dato filtrado
df = preparar_sesiones(
    df_filt,
    session_timeout=session_timeout,
    modo_sesion=modo_sesion,
)

# -----------------------------------------------------------
# CÁLCULO DE MÉTRICAS
# -----------------------------------------------------------

# Usuarios activos (empresa+usuario) en el periodo filtrado
usuarios_activos = df[["empresa", "usuario"]].drop_duplicates().shape[0]

# Sesiones y acciones
acciones_por_sesion = df.groupby("session_id")["function_used"].count()

# Frecuencia diaria básica (la usamos como base para diario/semanal/mensual)
sesiones_por_dia = (
    df.groupby("fecha")["session_id"]
    .nunique()
    .rename("Sesiones")
    .to_frame()
)
sesiones_por_dia.index = pd.to_datetime(sesiones_por_dia.index)  # para resample

# Tabla a nivel sesión (para conversión)
sesiones = df.groupby("session_id").agg(
    vio_producto=("es_ver", "max"),
    descargo_fds=("es_fds", "max"),
    descargo_etq=("es_etq", "max"),
)
sesiones["descargo_algo"] = sesiones["descargo_fds"] | sesiones["descargo_etq"]

sesiones_con_ver = sesiones[sesiones["vio_producto"]]

conv_fds = (
    sesiones_con_ver["descargo_fds"].mean()
    if not sesiones_con_ver.empty
    else 0.0
)
conv_etq = (
    sesiones_con_ver["descargo_etq"].mean()
    if not sesiones_con_ver.empty
    else 0.0
)
conv_cualquier = (
    sesiones_con_ver["descargo_algo"].mean()
    if not sesiones_con_ver.empty
    else 0.0
)

# Métricas por empresa (a nivel empresa+usuario+session_id)
sesiones_emp = df.groupby(["empresa", "usuario", "session_id"]).agg(
    eventos=("function_used", "count"),
    vio=("es_ver", "max"),
    fds=("es_fds", "max"),
    etq=("es_etq", "max"),
)

emp = sesiones_emp.groupby("empresa").agg(
    sesiones=("eventos", "size"),
    eventos_tot=("eventos", "sum"),
    profundidad=("eventos", "mean"),
    conv_fds=("fds", "mean"),
    conv_etq=("etq", "mean"),
).fillna(0)

emp["conv_prom"] = (emp["conv_fds"] + emp["conv_etq"]) / 2

# Normalización para índice de usabilidad
for col in ["sesiones", "profundidad", "conv_prom"]:
    if emp[col].max() > emp[col].min():
        emp[col + "_norm"] = (emp[col] - emp[col].min()) / (
            emp[col].max() - emp[col].min()
        )
    else:
        emp[col + "_norm"] = 0.0

emp["indice"] = (
    0.4 * emp["sesiones_norm"]
    + 0.3 * emp["profundidad_norm"]
    + 0.3 * emp["conv_prom_norm"]
)

# -----------------------------------------------------------
# KPIs PRINCIPALES
# -----------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)

col1.markdown(
    f"<div class='big-metric'>{emp.shape[0]}</div>"
    "<div class='metric-label'>Empresas activas</div>",
    unsafe_allow_html=True,
)
col2.markdown(
    f"<div class='big-metric'>{usuarios_activos}</div>"
    "<div class='metric-label'>Usuarios activos (empresa + usuario)</div>",
    unsafe_allow_html=True,
)
col3.markdown(
    f"<div class='big-metric'>{len(acciones_por_sesion)}</div>"
    "<div class='metric-label'>Sesiones en el periodo</div>",
    unsafe_allow_html=True,
)
col4.markdown(
    f"<div class='big-metric'>{acciones_por_sesion.mean():.1f}</div>"
    "<div class='metric-label'>Acciones promedio por sesión</div>",
    unsafe_allow_html=True,
)
col5.markdown(
    f"<div class='big-metric'>{conv_cualquier*100:.1f}%</div>"
    "<div class='metric-label'>Conversión ver → cualquier descarga</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='small-help'>Todas las métricas ya consideran los filtros de empresa, fechas y definición de sesión.</div>",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------
# PESTAÑAS
# -----------------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    [
        "📈 Uso en el tiempo y profundidad",
        "🎯 Conversión y flujos",
        "🏢 Top de empresas",
    ]
)

# -----------------------------------------------------------
# TAB 1 – FRECUENCIA Y PROFUNDIDAD
# -----------------------------------------------------------
with tab1:
    st.subheader("Frecuencia de uso – sesiones por periodo")

    nivel_tiempo = st.radio(
        "Nivel de detalle del tiempo",
        ["Promedio diario", "Promedio semanal", "Promedio mensual"],
        horizontal=True,
        help="Cambia la escala para ver la tendencia de uso.",
    )

    # Seleccionar granularidad
    s = sesiones_por_dia["Sesiones"]
    if nivel_tiempo == "Promedio diario":
        freq = s.to_frame("Sesiones promedio")
    elif nivel_tiempo == "Promedio semanal":
        freq = s.resample("W").mean().to_frame("Sesiones promedio")
    else:  # mensual
        freq = s.resample("M").mean().to_frame("Sesiones promedio")

    freq = freq.reset_index().rename(columns={"index": "Periodo"})
    freq.columns = ["Periodo", "Sesiones promedio"]

    if freq.empty:
        st.info("No hay sesiones en el periodo seleccionado.")
    else:
        chart_freq = (
            alt.Chart(freq)
            .mark_line(point=True)
            .encode(
                x=alt.X("Periodo:T", title="Fecha"),
                y=alt.Y("Sesiones promedio:Q", title="Sesiones promedio"),
                tooltip=[
                    alt.Tooltip("Periodo:T", title="Periodo"),
                    alt.Tooltip(
                        "Sesiones promedio:Q",
                        title="Sesiones promedio",
                        format=".2f",
                    ),
                ],
            )
            .properties(height=320)
            .interactive()
        )
        st.altair_chart(chart_freq, use_container_width=True)

    st.subheader("Profundidad de uso – distribución de acciones por sesión")

    depth_df = acciones_por_sesion.reset_index()
    depth_df.columns = ["ID sesión", "Número de acciones"]
    depth_df["Rango de acciones"] = depth_df["Número de acciones"].apply(
        bucket_acciones
    )

    depth_count = (
        depth_df.groupby("Rango de acciones")["ID sesión"]
        .count()
        .reset_index(name="Número de sesiones")
    )

    depth_chart = (
        alt.Chart(depth_count)
        .mark_bar()
        .encode(
            x=alt.X(
                "Rango de acciones:N",
                title="Rango de acciones en la sesión",
                sort=[
                    "1 acción",
                    "2–3 acciones",
                    "4–5 acciones",
                    "6–10 acciones",
                    "11+ acciones",
                ],
            ),
            y=alt.Y("Número de sesiones:Q", title="Cantidad de sesiones"),
            tooltip=[
                alt.Tooltip("Rango de acciones:N", title="Rango"),
                alt.Tooltip("Número de sesiones:Q", title="Sesiones"),
            ],
        )
        .properties(height=300)
    )

    st.altair_chart(depth_chart, use_container_width=True)

    with st.expander("Ver detalle de sesiones y profundidad"):
        st.dataframe(
            depth_df.sort_values("Número de acciones", ascending=False)
        )

# -----------------------------------------------------------
# TAB 2 – CONVERSIÓN Y FLUJOS
# -----------------------------------------------------------
with tab2:
    st.subheader("Tasa de conversión de tareas clave")

    conv_df = pd.DataFrame(
        {
            "Flujo": ["Ver → FDS", "Ver → Etiquetas", "Ver → cualquier descarga"],
            "Tasa de conversión": [conv_fds, conv_etq, conv_cualquier],
            "Descripción": [
                "Sesiones donde se vio un producto y se descargó al menos una FDS",
                "Sesiones donde se vio un producto y se descargó al menos una etiqueta",
                "Sesiones donde se vio un producto y se descargó FDS o etiqueta (lo que sea)",
            ],
        }
    )

    conv_chart = (
        alt.Chart(conv_df)
        .mark_bar(size=60)
        .encode(
            x=alt.X("Flujo:N", title="Flujo"),
            y=alt.Y(
                "Tasa de conversión:Q",
                title="Tasa de conversión",
                axis=alt.Axis(format="%"),
            ),
            tooltip=[
                alt.Tooltip("Flujo:N", title="Flujo"),
                alt.Tooltip(
                    "Tasa de conversión:Q", title="Conversión", format=".1%"
                ),
                alt.Tooltip("Descripción:N", title="Descripción"),
            ],
        )
        .properties(height=300)
    )

    st.altair_chart(conv_chart, use_container_width=True)

    st.markdown(
        """
Estas tasas se calculan **solo sobre sesiones donde hubo visualización del detalle de un producto**.
        """
    )

    st.subheader("Flujos más frecuentes (pares de acciones consecutivas)")

    df_sorted = df.sort_values(["session_id", "created_at"])
    df_sorted["accion_siguiente"] = df_sorted.groupby("session_id")[
        "function_used"
    ].shift(-1)

    rutas = (
        df_sorted.dropna(subset=["accion_siguiente"])
        .groupby(["function_used", "accion_siguiente"])
        .size()
        .reset_index(name="conteo")
        .sort_values("conteo", ascending=False)
        .head(15)
    )

    if rutas.empty:
        st.info("No hay suficientes secuencias de acciones para mostrar flujos.")
    else:
        rutas["Ruta"] = (
            rutas["function_used"] + " → " + rutas["accion_siguiente"]
        )

        rutas_chart = (
            alt.Chart(rutas)
            .mark_bar()
            .encode(
                y=alt.Y(
                    "Ruta:N",
                    sort="-x",
                    title="Ruta (acción actual → siguiente acción)",
                ),
                x=alt.X("conteo:Q", title="Veces que ocurre"),
                tooltip=[
                    alt.Tooltip("function_used:N", title="Acción actual"),
                    alt.Tooltip(
                        "accion_siguiente:N", title="Acción siguiente"
                    ),
                    alt.Tooltip("conteo:Q", title="Ocurrencias"),
                ],
            )
            .properties(height=400)
        )

        st.altair_chart(rutas_chart, use_container_width=True)

        with st.expander("Ver tabla de flujos detallada"):
            st.dataframe(
                rutas.rename(
                    columns={
                        "function_used": "Acción actual",
                        "accion_siguiente": "Acción siguiente",
                        "conteo": "Número de veces",
                    }
                )[["Acción actual", "Acción siguiente", "Número de veces"]]
            )

# -----------------------------------------------------------
# TAB 3 – TOP EMPRESAS
# -----------------------------------------------------------
with tab3:
    st.subheader("Ranking de empresas por métrica")

    opciones_ranking = {
        "Índice de usabilidad (compuesto)": "indice",
        "Número de sesiones": "sesiones",
        "Profundidad (acciones promedio por sesión)": "profundidad",
        "Conversión promedio (FDS + etiquetas)": "conv_prom",
    }

    metrica_label = st.selectbox(
        "Métrica para ordenar el ranking",
        list(opciones_ranking.keys()),
        index=0,
        help="El índice de usabilidad mezcla sesiones, profundidad y conversión.",
    )
    metrica_col = opciones_ranking[metrica_label]

    top_n = st.slider(
        "Número de empresas a mostrar", 5, 30, 15
    )

    emp_rank = (
        emp.sort_values(metrica_col, ascending=False)
        .head(top_n)
        .reset_index()
    )

    rank_chart = (
        alt.Chart(emp_rank)
        .mark_bar()
        .encode(
            y=alt.Y("empresa:N", sort="-x", title="Empresa"),
            x=alt.X(
                f"{metrica_col}:Q",
                title=metrica_label,
            ),
            tooltip=[
                alt.Tooltip("empresa:N", title="Empresa"),
                alt.Tooltip("sesiones:Q", title="Sesiones"),
                alt.Tooltip(
                    "profundidad:Q",
                    title="Acciones prom. por sesión",
                    format=".2f",
                ),
                alt.Tooltip(
                    "conv_prom:Q",
                    title="Conv. promedio (FDS+etq)",
                    format=".1%",
                ),
                alt.Tooltip(
                    "indice:Q",
                    title="Índice de usabilidad",
                    format=".2f",
                ),
            ],
        )
        .properties(height=500)
    )

    st.altair_chart(rank_chart, use_container_width=True)

    st.markdown("### Tabla de ranking")
    tabla_ranking = emp_rank.rename(
        columns={
            "empresa": "Empresa",
            "sesiones": "Sesiones",
            "profundidad": "Acciones prom. por sesión",
            "conv_prom": "Conv. promedio (FDS+etq)",
            "indice": "Índice de usabilidad",
        }
    )[["Empresa", "Sesiones", "Acciones prom. por sesión", "Conv. promedio (FDS+etq)", "Índice de usabilidad"]]

    st.dataframe(tabla_ranking)

    st.markdown("### Ficha rápida por empresa")

    empresa_focus = st.selectbox(
        "Selecciona una empresa",
        tabla_ranking["Empresa"].tolist(),
    )

    if empresa_focus:
        fila = emp.loc[empresa_focus]
        st.markdown(
            f"""
**{empresa_focus} – Ficha de usabilidad**

- Sesiones en el periodo: **{fila['sesiones']}**
- Acciones promedio por sesión: **{fila['profundidad']:.2f}**
- Conversión ver → FDS: **{fila['conv_fds']*100:.1f}%**
- Conversión ver → Etiquetas: **{fila['conv_etq']*100:.1f}%**
- Conversión promedio (FDS + Etiquetas): **{fila['conv_prom']*100:.1f}%**
- Índice de usabilidad (0–1): **{fila['indice']:.2f}**
            """
        )
