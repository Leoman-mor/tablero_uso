# app.py
# -----------------------------------------------------------
# Dashboard de Usabilidad Soluquim
#   - empresas.csv: uso de la plataforma
#   - productos.csv: creación / actualización de productos
#   - 5 pestañas:
#       1) Uso en el tiempo y profundidad
#       2) Conversión y flujos
#       3) Ranking de empresas (índice de usabilidad)
#       4) Tiempo entre acciones
#       5) Productos
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
def load_empresas():
    df = pd.read_csv("empresas.csv")
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["fecha"] = df["created_at"].dt.date
    df = df.sort_values(["empresa", "usuario", "created_at"])
    return df

def load_productos():
    dfp = pd.read_csv("productos.csv")
    dfp["created_at"] = pd.to_datetime(dfp["created_at"], errors="coerce")
    dfp["updated_at"] = pd.to_datetime(dfp["updated_at"], errors="coerce")
    # opcional: quitar columna índice del CSV
    if "Unnamed: 0" in dfp.columns:
        dfp = dfp.drop(columns=["Unnamed: 0"])
    return dfp


df_raw = load_empresas()
df_prod_raw = load_productos()

# -----------------------------------------------------------
# FUNCIONES AUXILIARES
# -----------------------------------------------------------
def preparar_sesiones(
    df: pd.DataFrame,
    modo_sesion: str = "login",
    session_timeout: int = 30,
) -> pd.DataFrame:
    """
    Crea session_id por empresa+usuario.

    - modo_sesion = 'login':
        Sesión = desde un evento 'Ingreso a login' hasta el siguiente
        para ese (empresa, usuario).
    - modo_sesion = 'inactividad':
        Sesión = bloque de acciones del mismo (empresa, usuario)
        sin pausas mayores a session_timeout minutos.

    El tiempo de inactividad se calcula por usuario en cada empresa.
    """
    df = df.copy().sort_values(["empresa", "usuario", "created_at"])
    group_keys = ["empresa", "usuario"]

    if modo_sesion == "login":
        df["es_ingreso"] = df["function_used"].str.contains(
            "Ingreso a login", case=False, na=False
        )
        df["sesion_local"] = df.groupby(group_keys)["es_ingreso"].cumsum()
        df["sesion_local"] = df["sesion_local"].where(df["sesion_local"] > 0, 1)

    elif modo_sesion == "inactividad":
        df["diff_min_tmp"] = (
            df.groupby(group_keys)["created_at"]
              .diff()
              .dt.total_seconds()
              .div(60)
        )
        df["nueva_sesion"] = df["diff_min_tmp"].isna() | (df["diff_min_tmp"] > session_timeout)
        df["sesion_local"] = df.groupby(group_keys)["nueva_sesion"].cumsum()
        df.drop(columns=["diff_min_tmp", "nueva_sesion"], inplace=True)
    else:
        raise ValueError("modo_sesion debe ser 'login' o 'inactividad'")

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


def minmax(col: pd.Series) -> pd.Series:
    """Normalización min–max a rango 0–1."""
    vmin, vmax = col.min(), col.max()
    if vmax > vmin:
        return (col - vmin) / (vmax - vmin)
    else:
        return pd.Series(0.0, index=col.index)

# -----------------------------------------------------------
# TÍTULO Y GLOSARIO
# -----------------------------------------------------------
st.title("📊 Dashboard de Usabilidad – Soluquim")
st.caption(
    "Analiza cómo las empresas usan Soluquim: frecuencia, profundidad, conversión, tiempos entre acciones y trabajo sobre productos."
)

with st.expander("ℹ️ Glosario rápido"):
    st.markdown(
        """
- **Sesión** (empresa + usuario):
  - Por **inicio de sesión**: desde un `Ingreso a login` hasta el siguiente.
  - Por **inactividad**: bloque continuo de actividad sin pausas mayores al umbral.
- **Frecuencia:** número de sesiones que abre una empresa.
- **Profundidad:** acciones promedio por sesión.
- **Conversión:** de las sesiones donde se ve un producto, porcentaje en que hay descargas (FDS o etiquetas).
- **Índice de usabilidad (0–1):** combina frecuencia (40%), profundidad (30%) y conversión (30%), ajustado por volumen de sesiones.
- **Tiempo entre acciones:** tiempo promedio (minutos) entre un evento y el siguiente dentro de cada sesión.
- **Productos trabajados:** conteo de productos distintos (id) creados o actualizados en un periodo. Una actualización se cuenta una sola vez por combinación (id, usuario).
- **Productos creados:** cantidad de productos distintos cuya fecha de creación está en el año seleccionado y que tienen un usuario de creación definido. Cada producto se cuenta una sola vez.
- **Productos actualizados:** cantidad de productos distintos con al menos una actualización en el año seleccionado y con usuario de actualización definido. También se cuentan una sola vez por producto.
- **Actividad por usuario:** número de productos en los que cada usuario aparece como creador o actualizador (deduplicado por producto). Solo se muestran usuarios con actividad y se limita al Top 20.
        """
    )

# -----------------------------------------------------------
# SIDEBAR – FILTROS (para datos de uso)
# -----------------------------------------------------------
st.sidebar.header("🎯 Alcance del análisis de uso")

# 1) Rango de fechas
min_fecha = df_raw["fecha"].min()
max_fecha = df_raw["fecha"].max()

rango_fechas = st.sidebar.date_input(
    "Periodo de análisis",
    value=(min_fecha, max_fecha),
    min_value=min_fecha,
    max_value=max_fecha,
    help="Solo se consideran eventos dentro de este rango.",
)

if isinstance(rango_fechas, tuple):
    fecha_ini, fecha_fin = rango_fechas
else:
    fecha_ini, fecha_fin = rango_fechas, rango_fechas

# 2) Empresa (una o todas)
empresas_unicas = sorted(df_raw["empresa"].unique())
opciones_emp = ["Todas las empresas"] + empresas_unicas
empresa_sel = st.sidebar.selectbox(
    "Empresa",
    options=opciones_emp,
    help="Aplica a las pestañas 1–4 (uso). La pestaña de productos se analiza a nivel global.",
)

if empresa_sel == "Todas las empresas":
    empresas_sel = empresas_unicas
else:
    empresas_sel = [empresa_sel]

st.sidebar.markdown("---")

# 3) Sesiones: login vs inactividad por usuario
st.sidebar.subheader("⚙️ Sesiones")

modo_sesion_label = st.sidebar.radio(
    "¿Cómo quieres contar una sesión?",
    [
        "Por inicio de sesión (Ingreso a login)",
        "Por inactividad del usuario (X min sin eventos)",
    ],
)

if modo_sesion_label.startswith("Por inicio"):
    modo_sesion = "login"
    session_timeout = 30  # dummy
    st.sidebar.caption(
        "Sesión = desde un 'Ingreso a login' hasta el siguiente, para ese usuario en esa empresa."
    )
else:
    modo_sesion = "inactividad"
    session_timeout = st.sidebar.slider(
        "Umbral de inactividad por usuario (minutos)",
        min_value=5,
        max_value=90,
        value=30,
        step=5,
        help="Si un usuario pasa más de este tiempo sin eventos, se corta la sesión.",
    )
    st.sidebar.caption(
        "Sesión = bloque de acciones del mismo usuario sin pausas mayores a este tiempo."
    )

# -----------------------------------------------------------
# APLICAR FILTROS Y CONSTRUIR SESIONES (USO)
# -----------------------------------------------------------
df_filt = df_raw[
    (df_raw["empresa"].isin(empresas_sel))
    & (df_raw["fecha"].between(fecha_ini, fecha_fin))
].copy()

if df_filt.empty:
    st.warning("No hay datos de uso para la empresa y el periodo seleccionados.")
    st.stop()

df = preparar_sesiones(
    df_filt,
    modo_sesion=modo_sesion,
    session_timeout=session_timeout,
)

# -----------------------------------------------------------
# MÉTRICAS GLOBALES (USO)
# -----------------------------------------------------------
usuarios_activos = df[["empresa", "usuario"]].drop_duplicates().shape[0]
acciones_por_sesion = df.groupby("session_id")["function_used"].count()

# Frecuencia diaria (base para curvas)
sesiones_por_dia = (
    df.groupby("fecha")["session_id"]
    .nunique()
    .rename("Sesiones")
    .to_frame()
)
sesiones_por_dia.index = pd.to_datetime(sesiones_por_dia.index)

# Conversión global
sesiones = df.groupby("session_id").agg(
    vio_producto=("es_ver", "max"),
    descargo_fds=("es_fds", "max"),
    descargo_etq=("es_etq", "max"),
)
sesiones["descargo_algo"] = sesiones["descargo_fds"] | sesiones["descargo_etq"]
sesiones_con_ver = sesiones[sesiones["vio_producto"]]

conv_fds = sesiones_con_ver["descargo_fds"].mean() if not sesiones_con_ver.empty else 0.0
conv_etq = sesiones_con_ver["descargo_etq"].mean() if not sesiones_con_ver.empty else 0.0
conv_cualquier = (
    sesiones_con_ver["descargo_algo"].mean() if not sesiones_con_ver.empty else 0.0
)

# -----------------------------------------------------------
# MÉTRICAS POR EMPRESA E ÍNDICE DE USABILIDAD
# -----------------------------------------------------------
sesiones_emp = df.groupby(["empresa", "usuario", "session_id"]).agg(
    eventos=("function_used", "count"),
    vio=("es_ver", "max"),
    fds=("es_fds", "max"),
    etq=("es_etq", "max"),
)

emp = sesiones_emp.groupby("empresa").agg(
    sesiones=("eventos", "size"),      # número de sesiones
    eventos_tot=("eventos", "sum"),    # acciones totales
    profundidad=("eventos", "mean"),   # acciones promedio por sesión
    conv_fds=("fds", "mean"),
    conv_etq=("etq", "mean"),
).fillna(0)

emp["conv_prom"] = (emp["conv_fds"] + emp["conv_etq"]) / 2

emp["freq_norm"] = minmax(emp["sesiones"])
emp["prof_norm"] = minmax(emp["profundidad"])
emp["conv_norm"] = minmax(emp["conv_prom"])

w_freq, w_prof, w_conv = 0.4, 0.3, 0.3
emp["indice_base"] = (
    w_freq * emp["freq_norm"]
    + w_prof * emp["prof_norm"]
    + w_conv * emp["conv_norm"]
)

SESIONES_CONFIABLES = 10  # a partir de 10 sesiones no hay castigo
emp["factor_volumen"] = (
    emp["sesiones"].clip(upper=SESIONES_CONFIABLES) / SESIONES_CONFIABLES
)

emp["indice_usabilidad"] = emp["indice_base"] * emp["factor_volumen"]

# -----------------------------------------------------------
# PESTAÑAS
# -----------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📈 Uso en el tiempo y profundidad",
        "🎯 Conversión y flujos",
        "🏢 Ranking de empresas",
        "⏱️ Tiempo entre acciones",
        "📦 Productos",
    ]
)

# -----------------------------------------------------------
# TAB 1 – FRECUENCIA Y PROFUNDIDAD
# -----------------------------------------------------------
with tab1:
    st.subheader("Visión general de uso")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.markdown(
        f"<div class='big-metric'>{emp.shape[0]}</div>"
        "<div class='metric-label'>Empresas con actividad en el periodo</div>",
        unsafe_allow_html=True,
    )
    col2.markdown(
        f"<div class='big-metric'>{usuarios_activos}</div>"
        "<div class='metric-label'>Usuarios activos (empresa + usuario)</div>",
        unsafe_allow_html=True,
    )
    col3.markdown(
        f"<div class='big-metric'>{len(acciones_por_sesion)}</div>"
        "<div class='metric-label'>Sesiones (según definición seleccionada)</div>",
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
        "<div class='small-help'>Estas métricas respetan los filtros de empresa, fechas y definición de sesión.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.subheader("Frecuencia de uso – sesiones por periodo")

    nivel_tiempo = st.radio(
        "Nivel de detalle del tiempo",
        ["Promedio diario", "Promedio semanal", "Promedio mensual"],
        horizontal=True,
    )

    s = sesiones_por_dia["Sesiones"]
    if nivel_tiempo == "Promedio diario":
        freq = s.to_frame("Sesiones promedio")
    elif nivel_tiempo == "Promedio semanal":
        freq = s.resample("W").mean().to_frame("Sesiones promedio")
    else:
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

    with st.expander("Detalle de sesiones y profundidad"):
        st.dataframe(depth_df.sort_values("Número de acciones", ascending=False))

# -----------------------------------------------------------
# TAB 2 – CONVERSIÓN Y FLUJOS
# -----------------------------------------------------------
with tab2:

    st.subheader("Visión general de uso")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.markdown(
        f"<div class='big-metric'>{emp.shape[0]}</div>"
        "<div class='metric-label'>Empresas con actividad en el periodo</div>",
        unsafe_allow_html=True,
    )
    col2.markdown(
        f"<div class='big-metric'>{usuarios_activos}</div>"
        "<div class='metric-label'>Usuarios activos (empresa + usuario)</div>",
        unsafe_allow_html=True,
    )
    col3.markdown(
        f"<div class='big-metric'>{len(acciones_por_sesion)}</div>"
        "<div class='metric-label'>Sesiones (según definición seleccionada)</div>",
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
        "<div class='small-help'>Estas métricas respetan los filtros de empresa, fechas y definición de sesión.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.subheader("Tasa de conversión de tareas clave")

    conv_df = pd.DataFrame(
        {
            "Flujo": ["Ver → FDS", "Ver → Etiquetas", "Ver → cualquier descarga"],
            "Tasa de conversión": [conv_fds, conv_etq, conv_cualquier],
            "Descripción": [
                "Sesiones donde se vio un producto y se descargó al menos una FDS",
                "Sesiones donde se vio un producto y se descargó al menos una etiqueta",
                "Sesiones donde se vio un producto y se descargó FDS o etiqueta",
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
        "Estas tasas se calculan **solo sobre sesiones donde hubo visualización del detalle de un producto**."
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

        with st.expander("Tabla de flujos detallada"):
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
# TAB 3 – RANKING DE EMPRESAS
# -----------------------------------------------------------
with tab3:
    st.subheader("Visión general de uso")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.markdown(
        f"<div class='big-metric'>{emp.shape[0]}</div>"
        "<div class='metric-label'>Empresas con actividad en el periodo</div>",
        unsafe_allow_html=True,
    )
    col2.markdown(
        f"<div class='big-metric'>{usuarios_activos}</div>"
        "<div class='metric-label'>Usuarios activos (empresa + usuario)</div>",
        unsafe_allow_html=True,
    )
    col3.markdown(
        f"<div class='big-metric'>{len(acciones_por_sesion)}</div>"
        "<div class='metric-label'>Sesiones (según definición seleccionada)</div>",
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
        "<div class='small-help'>Estas métricas respetan los filtros de empresa, fechas y definición de sesión.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.subheader("Ranking de empresas por usabilidad")

    # Filtro interno fijo: mínimo 3 sesiones
    SESIONES_MINIMAS = 3
    emp_filtrado = emp[emp["sesiones"] >= SESIONES_MINIMAS]

    opciones_ranking = {
        "Índice de usabilidad (0–1)": "indice_usabilidad",
        "Índice base (sin ajuste por volumen)": "indice_base",
        "Número de sesiones": "sesiones",
        "Acciones promedio por sesión": "profundidad",
        "Conversión promedio (FDS + etiquetas)": "conv_prom",
    }

    metrica_label = st.selectbox(
        "Métrica para ordenar el ranking",
        list(opciones_ranking.keys()),
        index=0,
    )
    metrica_col = opciones_ranking[metrica_label]

    top_n = st.slider("Número de empresas a mostrar", 5, 30, 15)

    if emp_filtrado.empty:
        st.info("No hay empresas que cumplan el mínimo de sesiones (≥ 3).")
    else:
        emp_rank = (
            emp_filtrado.sort_values(metrica_col, ascending=False)
            .head(top_n)
            .reset_index()
        )

        rank_chart = (
            alt.Chart(emp_rank)
            .mark_bar()
            .encode(
                y=alt.Y("empresa:N", sort="-x", title="Empresa"),
                x=alt.X(f"{metrica_col}:Q", title=metrica_label),
                tooltip=[
                    alt.Tooltip("empresa:N", title="Empresa"),
                    alt.Tooltip("sesiones:Q", title="Sesiones"),
                    alt.Tooltip("eventos_tot:Q", title="Acciones totales"),
                    alt.Tooltip(
                        "profundidad:Q",
                        title="Acciones promedio por sesión",
                        format=".2f",
                    ),
                    alt.Tooltip(
                        "conv_prom:Q",
                        title="Conv. promedio (FDS+etq)",
                        format=".1%",
                    ),
                    alt.Tooltip(
                        "indice_base:Q",
                        title="Índice base (0–1)",
                        format=".2f",
                    ),
                    alt.Tooltip(
                        "indice_usabilidad:Q",
                        title="Índice de usabilidad (0–1)",
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
                "eventos_tot": "Acciones totales",
                "profundidad": "Acciones prom. por sesión",
                "conv_prom": "Conv. promedio (FDS+etq)",
                "indice_base": "Índice base (0–1)",
                "indice_usabilidad": "Índice de usabilidad (0–1)",
            }
        )[
            [
                "Empresa",
                "Sesiones",
                "Acciones totales",
                "Acciones prom. por sesión",
                "Conv. promedio (FDS+etq)",
                "Índice base (0–1)",
                "Índice de usabilidad (0–1)",
            ]
        ]

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
- Acciones totales: **{fila['eventos_tot']}**
- Acciones promedio por sesión: **{fila['profundidad']:.2f}**
- Conversión ver → FDS: **{fila['conv_fds']*100:.1f}%**
- Conversión ver → Etiquetas: **{fila['conv_etq']*100:.1f}%**
- Conversión promedio (FDS + Etiquetas): **{fila['conv_prom']*100:.1f}%**
- Índice base (0–1): **{fila['indice_base']:.2f}**
- Índice de usabilidad (0–1): **{fila['indice_usabilidad']:.2f}**
                """
            )

# -----------------------------------------------------------
# TAB 4 – TIEMPOS PARA DESCARGAR FDS / ETIQUETAS
# -----------------------------------------------------------
with tab4:
    st.subheader("⏱️ Tiempos para llegar a descargas de FDS / etiquetas")

    st.markdown(
        "<div class='small-help'>"
        "Aquí analizamos cuánto tardan las empresas, en promedio, en generar una descarga "
        "de FDS o de etiquetas desde que sus usuarios entran al módulo de productos. "
        "Sirve para identificar empresas ágiles y empresas que pueden requerir acompañamiento "
        "en el uso de Soluquim."
        "</div>",
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------
    # 1. Construcción de tiempos por sesión
    #    Punto de inicio: Acceso al módulo de productos
    # -------------------------------------------------------
    df_ses = df.sort_values(["session_id", "created_at"]).copy()

    # Sesiones base: empresa, usuario, inicio y fin reales
    ses_base = (
        df_ses.groupby("session_id")
        .agg(
            empresa=("empresa", "first"),
            usuario=("usuario", "first"),
            inicio_sesion=("created_at", "min"),
            fin_sesion=("created_at", "max"),
        )
        .reset_index()
    )

    # Primer acceso al módulo de productos
    mod_times = (
        df_ses[df_ses["function_used"] == "Acceso al módulo de productos"]
        .groupby("session_id")["created_at"]
        .min()
        .rename("t_modulo_productos")
    )

    # Primera descarga de FDS y de etiquetas
    fds_times = (
        df_ses[df_ses["es_fds"]]
        .groupby("session_id")["created_at"]
        .min()
        .rename("t_fds")
    )
    etq_times = (
        df_ses[df_ses["es_etq"]]
        .groupby("session_id")["created_at"]
        .min()
        .rename("t_etq")
    )

    # (Opcional) primera visualización de detalle, como respaldo
    ver_times = (
        df_ses[df_ses["es_ver"]]
        .groupby("session_id")["created_at"]
        .min()
        .rename("t_ver")
    )

    # Unimos todo
    ses = (
        ses_base
        .merge(mod_times, on="session_id", how="left")
        .merge(ver_times, on="session_id", how="left")
        .merge(fds_times, on="session_id", how="left")
        .merge(etq_times, on="session_id", how="left")
    )

    # Punto de inicio del análisis de tiempos:
    # 1) Si existe "Acceso al módulo de productos", usamos ese tiempo
    # 2) Si no, usamos primera visualización de producto (t_ver)
    # 3) Si tampoco hay, usamos inicio de la sesión
    ses["t_inicio_trabajo"] = (
        ses["t_modulo_productos"]
        .combine_first(ses["t_ver"])
        .combine_first(ses["inicio_sesion"])
    )

    # Función auxiliar para obtener diferencias en minutos
    def diff_minutos(col_fin: str) -> pd.Series:
        return (
            (ses[col_fin] - ses["t_inicio_trabajo"])
            .dt.total_seconds()
            .div(60)
        )

    # Tiempos en minutos hasta cada tipo de descarga
    ses["min_hasta_fds"] = diff_minutos("t_fds")
    ses["min_hasta_etq"] = diff_minutos("t_etq")

    # Primera descarga de cualquier tipo (FDS o etiqueta)
    ses["t_desc_cualquiera"] = ses[["t_fds", "t_etq"]].min(axis=1)
    ses["min_hasta_desc_cualquiera"] = diff_minutos("t_desc_cualquiera")

    total_sesiones = ses["session_id"].nunique()
    if total_sesiones == 0:
        st.info("No hay sesiones en el periodo y filtros seleccionados.")
        st.stop()

    # Sólo consideramos tiempos razonables (0–120 min)
    MAX_MIN = 120

    mask_fds = (
        ses["min_hasta_fds"].notna()
        & (ses["min_hasta_fds"] >= 0)
        & (ses["min_hasta_fds"] <= MAX_MIN)
    )
    mask_etq = (
        ses["min_hasta_etq"].notna()
        & (ses["min_hasta_etq"] >= 0)
        & (ses["min_hasta_etq"] <= MAX_MIN)
    )
    mask_any = (
        ses["min_hasta_desc_cualquiera"].notna()
        & (ses["min_hasta_desc_cualquiera"] >= 0)
        & (ses["min_hasta_desc_cualquiera"] <= MAX_MIN)
    )

    if not mask_fds.any() and not mask_etq.any():
        st.info(
            "No se encontraron sesiones que lleguen a descargar FDS o etiquetas "
            "en el periodo y filtros seleccionados."
        )
        st.stop()

    # -------------------------------------------------------
    # 2. Métricas globales de tiempos
    # -------------------------------------------------------
    sesiones_con_fds = ses.loc[mask_fds, "session_id"].nunique()
    sesiones_con_etq = ses.loc[mask_etq, "session_id"].nunique()
    sesiones_con_desc_any = ses.loc[mask_any, "session_id"].nunique()
    empresas_con_desc = ses.loc[mask_any, "empresa"].nunique()

    diffs_fds = ses.loc[mask_fds, "min_hasta_fds"]
    diffs_etq = ses.loc[mask_etq, "min_hasta_etq"]
    diffs_any = ses.loc[mask_any, "min_hasta_desc_cualquiera"]

    prom_fds = diffs_fds.mean() if not diffs_fds.empty else np.nan
    prom_etq = diffs_etq.mean() if not diffs_etq.empty else np.nan
    prom_any = diffs_any.mean() if not diffs_any.empty else np.nan

    col_a, col_b, col_c, col_d = st.columns(4)

    col_a.markdown(
        f"<div class='big-metric'>{empresas_con_desc}</div>"
        "<div class='metric-label'>Empresas con al menos una descarga</div>",
        unsafe_allow_html=True,
    )
    col_b.markdown(
        f"<div class='big-metric'>{sesiones_con_desc_any}</div>"
        f"<div class='metric-label'>Sesiones que llegan a descargar algo ({sesiones_con_desc_any/total_sesiones*100:.1f}% del total)</div>",
        unsafe_allow_html=True,
    )

    if not np.isnan(prom_fds):
        col_c.markdown(
            f"<div class='big-metric'>{prom_fds:.1f}</div>"
            "<div class='metric-label'>Minutos promedio hasta 1ª FDS</div>",
            unsafe_allow_html=True,
        )
    else:
        col_c.markdown(
            "<div class='big-metric'>–</div>"
            "<div class='metric-label'>Minutos promedio hasta 1ª FDS</div>",
            unsafe_allow_html=True,
        )

    if not np.isnan(prom_etq):
        col_d.markdown(
            f"<div class='big-metric'>{prom_etq:.1f}</div>"
            "<div class='metric-label'>Minutos promedio hasta 1ª etiqueta</div>",
            unsafe_allow_html=True,
        )
    else:
        col_d.markdown(
            "<div class='big-metric'>–</div>"
            "<div class='metric-label'>Minutos promedio hasta 1ª etiqueta</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='small-help'>"
        "Los tiempos se miden desde el primer ingreso al módulo de productos en la sesión; "
        "si no hay registro de ingreso al módulo, se usa la primera visualización de producto "
        "y, en última instancia, el inicio de la sesión. Solo se consideran tiempos entre 0 y 120 minutos "
        "para evitar casos atípicos extremos."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.subheader("🏢 Ranking de empresas por tiempo promedio")

    # -------------------------------------------------------
    # 3. Agregación por empresa
    # -------------------------------------------------------
    # Sesiones totales por empresa
    ses_tot_emp = (
        ses.groupby("empresa")["session_id"]
        .nunique()
        .reset_index(name="sesiones_totales")
    )

    # Métricas FDS por empresa
    fds_emp = (
        ses.loc[mask_fds]
        .groupby("empresa")["min_hasta_fds"]
        .agg(["mean", "median", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "tiempo_prom_fds",
                "median": "tiempo_mediano_fds",
                "count": "sesiones_con_fds",
            }
        )
    )

    # Métricas etiquetas por empresa
    etq_emp = (
        ses.loc[mask_etq]
        .groupby("empresa")["min_hasta_etq"]
        .agg(["mean", "median", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "tiempo_prom_etq",
                "median": "tiempo_mediano_etq",
                "count": "sesiones_con_etq",
            }
        )
    )

    # Consolidado
    ranking = (
        ses_tot_emp
        .merge(fds_emp, on="empresa", how="left")
        .merge(etq_emp, on="empresa", how="left")
    )

    # Empresas con al menos N sesiones con cada tipo de descarga
    MIN_SES_EMPRESA = 5

    ranking_fds = ranking[
        ranking["sesiones_con_fds"].fillna(0) >= MIN_SES_EMPRESA
    ].copy()
    ranking_etq = ranking[
        ranking["sesiones_con_etq"].fillna(0) >= MIN_SES_EMPRESA
    ].copy()

    num_emp_disp = max(ranking_fds.shape[0], ranking_etq.shape[0])

    if num_emp_disp == 0:
        st.info(
            "No hay empresas con suficientes sesiones que lleguen a descargar FDS o etiquetas "
            f"(se requieren al menos {MIN_SES_EMPRESA} sesiones con descarga por empresa)."
        )
    else:
        # Control de orden para ver rápidos o lentos primero
        orden_opcion = st.radio(
            "Orden del ranking",
            ["Más rápidos primero", "Más lentos primero"],
            horizontal=True,
            help="Elige si quieres ver primero las empresas más ágiles o las más lentas.",
        )
        asc = True if orden_opcion == "Más rápidos primero" else False
        sort_order = "ascending" if asc else "descending"

        top_n = st.slider(
            "Número de empresas a mostrar en el ranking",
            min_value=5,
            max_value=min(30, num_emp_disp),
            value=min(15, num_emp_disp),
            help="Controla cuántas empresas aparecen en las barras.",
        )

        # ---------------------------------------------------
        # 3.1. Gráfica FDS
        # ---------------------------------------------------
        col_fds, col_etq = st.columns(2)

        with col_fds:
            st.markdown("#### ⬇️ Tiempo promedio hasta la primera FDS")

            if ranking_fds.empty:
                st.info(
                    f"No hay empresas con al menos {MIN_SES_EMPRESA} sesiones que descarguen FDS."
                )
            else:
                data_fds = (
                    ranking_fds
                    .sort_values("tiempo_prom_fds", ascending=asc)
                    .head(top_n)
                    .reset_index(drop=True)
                )

                chart_fds = (
                    alt.Chart(data_fds)
                    .mark_bar()
                    .encode(
                        y=alt.Y(
                            "empresa:N",
                            title="Empresa",
                            sort=alt.SortField(
                                field="tiempo_prom_fds",
                                order=sort_order,
                            ),
                        ),
                        x=alt.X(
                            "tiempo_prom_fds:Q",
                            title="Minutos promedio hasta 1ª FDS",
                        ),
                        tooltip=[
                            alt.Tooltip("empresa:N", title="Empresa"),
                            alt.Tooltip(
                                "sesiones_totales:Q",
                                title="Sesiones totales",
                            ),
                            alt.Tooltip(
                                "sesiones_con_fds:Q",
                                title="Sesiones con FDS",
                            ),
                            alt.Tooltip(
                                "tiempo_prom_fds:Q",
                                title="Promedio (min)",
                                format=".2f",
                            ),
                            alt.Tooltip(
                                "tiempo_mediano_fds:Q",
                                title="Mediana (min)",
                                format=".2f",
                            ),
                        ],
                    )
                    .properties(height=450)
                )

                st.altair_chart(chart_fds, use_container_width=True)
                st.markdown(
                    "<div class='small-help'>"
                    "Según el orden seleccionado, verás primero las empresas más ágiles o las más lentas "
                    "para generar FDS desde el módulo de productos."
                    "</div>",
                    unsafe_allow_html=True,
                )

        # ---------------------------------------------------
        # 3.2. Gráfica Etiquetas
        # ---------------------------------------------------
        with col_etq:
            st.markdown("#### 🏷️ Tiempo promedio hasta la primera etiqueta")

            if ranking_etq.empty:
                st.info(
                    f"No hay empresas con al menos {MIN_SES_EMPRESA} sesiones que descarguen etiquetas."
                )
            else:
                data_etq = (
                    ranking_etq
                    .sort_values("tiempo_prom_etq", ascending=asc)
                    .head(top_n)
                    .reset_index(drop=True)
                )

                chart_etq = (
                    alt.Chart(data_etq)
                    .mark_bar()
                    .encode(
                        y=alt.Y(
                            "empresa:N",
                            title="Empresa",
                            sort=alt.SortField(
                                field="tiempo_prom_etq",
                                order=sort_order,
                            ),
                        ),
                        x=alt.X(
                            "tiempo_prom_etq:Q",
                            title="Minutos promedio hasta 1ª etiqueta",
                        ),
                        tooltip=[
                            alt.Tooltip("empresa:N", title="Empresa"),
                            alt.Tooltip(
                                "sesiones_totales:Q",
                                title="Sesiones totales",
                            ),
                            alt.Tooltip(
                                "sesiones_con_etq:Q",
                                title="Sesiones con etiquetas",
                            ),
                            alt.Tooltip(
                                "tiempo_prom_etq:Q",
                                title="Promedio (min)",
                                format=".2f",
                            ),
                            alt.Tooltip(
                                "tiempo_mediano_etq:Q",
                                title="Mediana (min)",
                                format=".2f",
                            ),
                        ],
                    )
                    .properties(height=450)
                )

                st.altair_chart(chart_etq, use_container_width=True)
                st.markdown(
                    "<div class='small-help'>"
                    "Según el orden seleccionado, verás primero las empresas que generan etiquetas más rápido "
                    "o las que tardan más."
                    "</div>",
                    unsafe_allow_html=True,
                )

    # -------------------------------------------------------
    # 4. Tabla detallada por empresa (modelo similar a otras pestañas)
    # -------------------------------------------------------
    st.markdown("### 📄 Tabla detallada de tiempos por empresa")

    tabla = ranking.rename(
        columns={
            "empresa": "Empresa",
            "sesiones_totales": "Sesiones totales",
            "sesiones_con_fds": "Sesiones con FDS",
            "sesiones_con_etq": "Sesiones con etiquetas",
            "tiempo_prom_fds": "Tiempo prom. hasta 1ª FDS (min)",
            "tiempo_mediano_fds": "Tiempo mediano hasta 1ª FDS (min)",
            "tiempo_prom_etq": "Tiempo prom. hasta 1ª etiqueta (min)",
            "tiempo_mediano_etq": "Tiempo mediano hasta 1ª etiqueta (min)",
        }
    )

    cols_tabla = [
        "Empresa",
        "Sesiones totales",
        "Sesiones con FDS",
        "Sesiones con etiquetas",
        "Tiempo prom. hasta 1ª FDS (min)",
        "Tiempo mediano hasta 1ª FDS (min)",
        "Tiempo prom. hasta 1ª etiqueta (min)",
        "Tiempo mediano hasta 1ª etiqueta (min)",
    ]
    cols_tabla = [c for c in cols_tabla if c in tabla.columns]

    # Usamos la misma lógica de orden (rápidos / lentos) que en las gráficas
    # Priorizamos ordenar por FDS; si no existe, por etiquetas
    sort_col_candidates = [
        "Tiempo prom. hasta 1ª FDS (min)",
        "Tiempo prom. hasta 1ª etiqueta (min)",
    ]
    sort_col = next((c for c in sort_col_candidates if c in tabla.columns), None)

    if sort_col:
        # asc viene definido arriba dentro del bloque "if num_emp_disp > 0"
        # Si no se llegó a ese bloque (caso extremo), asumimos asc = True por defecto
        if "asc" not in locals():
            asc = True

        tabla_sorted = tabla[cols_tabla].sort_values(
            sort_col,
            ascending=asc,
            na_position="last",
        )
    else:
        tabla_sorted = tabla[cols_tabla]

    st.dataframe(tabla_sorted)


# -----------------------------------------------------------
# TAB 5 – PRODUCTOS (CREACIÓN / ACTUALIZACIÓN)
# -----------------------------------------------------------
with tab5:
    st.subheader("📦 Productos creados y actualizados")

    prod = df_prod_raw.copy()

    # Limpiamos usuarios (quitamos espacios y nulos)
    prod["usuario_crea_clean"] = prod["usuario_crea"].fillna("").astype(str).str.strip()
    prod["usuario_actualiza_clean"] = prod["usuario_actualiza"].fillna("").astype(str).str.strip()

    # Años disponibles (por created o updated)
    years_created = prod["created_at"].dt.year.dropna().astype(int)
    years_updated = prod["updated_at"].dt.year.dropna().astype(int)
    years_all = sorted(set(years_created.tolist()) | set(years_updated.tolist()))

    # Filtro de año (solo para métricas y serie temporal)
    default_year = 2025 if 2025 in years_all else max(years_all)
    year = st.selectbox(
        "Año a analizar",
        years_all,
        index=years_all.index(default_year),
    )

    # Máscaras por año
    mask_created_year = prod["created_at"].dt.year == year
    mask_updated_year = prod["updated_at"].dt.year == year

    # Máscaras de usuarios válidos
    mask_crea_user_valido = prod["usuario_crea_clean"] != ""
    mask_act_user_valido = prod["usuario_actualiza_clean"] != ""

    # ---------------- MÉTRICAS GLOBALES ----------------
    total_productos = prod["id"].nunique()

    # Solo productos con responsable en ese año
    creados_ano = prod.loc[
        mask_created_year & mask_crea_user_valido, "id"
    ].nunique()

    actualizados_ano = prod.loc[
        mask_updated_year & mask_act_user_valido, "id"
    ].nunique()

    c1, c2, c3 = st.columns(3)
    c1.markdown(
        f"<div class='big-metric'>{total_productos}</div>"
        "<div class='metric-label'>Productos totales en Soluquim</div>",
        unsafe_allow_html=True,
    )
    c2.markdown(
        f"<div class='big-metric'>{creados_ano}</div>"
        f"<div class='metric-label'>Productos creados en {year}</div>",
        unsafe_allow_html=True,
    )
    c3.markdown(
        f"<div class='big-metric'>{actualizados_ano}</div>"
        f"<div class='metric-label'>Productos actualizados en {year}</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='small-help'>El total de productos incluye todos los ids. "
        "Las métricas por año se basan en created_at/updated_at de ese año, solo con usuarios definidos.</div>",
        unsafe_allow_html=True,
    )

    # ---------------- Serie mensual (creación + actualización en una sola gráfica) ----------------
    st.markdown("### 📈 Creación y actualización mensual de productos")

    # Solo contamos meses con responsable en ese año
    prod_year_created = prod.loc[
        mask_created_year & mask_crea_user_valido
    ].copy()
    prod_year_updated = prod.loc[
        mask_updated_year & mask_act_user_valido
    ].copy()

    if not prod_year_created.empty:
        prod_year_created["mes"] = prod_year_created["created_at"].dt.to_period("M").astype(str)
    if not prod_year_updated.empty:
        prod_year_updated["mes"] = prod_year_updated["updated_at"].dt.to_period("M").astype(str)

    serie_creados = (
        prod_year_created.groupby("mes")["id"].nunique().reset_index(name="creados")
        if not prod_year_created.empty
        else pd.DataFrame(columns=["mes", "creados"])
    )
    serie_actualizados = (
        prod_year_updated.groupby("mes")["id"].nunique().reset_index(name="actualizados")
        if not prod_year_updated.empty
        else pd.DataFrame(columns=["mes", "actualizados"])
    )

    serie = pd.merge(serie_creados, serie_actualizados, on="mes", how="outer").fillna(0)

    if not serie.empty:
        serie = serie.sort_values("mes")
        serie["mes_dt"] = pd.to_datetime(serie["mes"] + "-01")

        # Formato largo para una sola gráfica con dos líneas
        serie_long = serie.melt(
            id_vars=["mes", "mes_dt"],
            value_vars=["creados", "actualizados"],
            var_name="tipo",
            value_name="productos"
        )

        serie_long["tipo"] = serie_long["tipo"].replace(
            {
                "creados": "Creados",
                "actualizados": "Actualizados",
            }
        )

        chart_line = (
            alt.Chart(serie_long)
            .mark_line(point=True)
            .encode(
                x=alt.X("mes_dt:T", title="Mes"),
                y=alt.Y("productos:Q", title="Productos trabajados"),
                color=alt.Color("tipo:N", title="Tipo"),
                tooltip=[
                    alt.Tooltip("mes:N", title="Mes"),
                    alt.Tooltip("tipo:N", title="Tipo"),
                    alt.Tooltip("productos:Q", title="Productos"),
                ],
            )
            .properties(title=f"Productos creados y actualizados por mes en {year}", height=320)
        )

        st.altair_chart(chart_line, use_container_width=True)
    else:
        st.info(f"No hay productos trabajados (con responsable) en {year}.")

    # ---------------- Actividad por usuario (DINÁMICA) ----------------
    st.markdown("### 👥 Actividad por usuario (productos trabajados)")

    modo_usuarios = st.radio(
        "Periodo para el análisis por usuario",
        ["Solo año seleccionado", "Histórico completo"],
        horizontal=True,
        help="Elige si quieres ver la actividad solo para el año elegido o para toda la historia."
    )

    if modo_usuarios == "Solo año seleccionado":
        # Solo productos del año filtrado
        mask_crea_base = mask_created_year & mask_crea_user_valido
        mask_act_base = mask_updated_year & mask_act_user_valido
        sufijo_titulo = f"en {year}"
    else:
        # Todo el histórico (sin filtrar por año)
        mask_crea_base = mask_crea_user_valido
        mask_act_base = mask_act_user_valido
        sufijo_titulo = " (histórico completo)"

    # CREADOS: deduplicamos por (id, usuario_crea_clean), sin usuarios vacíos
    crea_user_df = (
        prod.loc[
            mask_crea_base,
            ["id", "usuario_crea_clean"],
        ]
        .drop_duplicates(subset=["id", "usuario_crea_clean"])
        .rename(columns={"usuario_crea_clean": "usuario"})
    )
    crea_user = (
        crea_user_df.groupby("usuario")["id"]
        .nunique()
        .reset_index(name="productos_creados")
    )

    # ACTUALIZADOS: deduplicamos por (id, usuario_actualiza_clean), sin usuarios vacíos
    act_user_df = (
        prod.loc[
            mask_act_base,
            ["id", "usuario_actualiza_clean"],
        ]
        .drop_duplicates(subset=["id", "usuario_actualiza_clean"])
        .rename(columns={"usuario_actualiza_clean": "usuario"})
    )
    act_user = (
        act_user_df.groupby("usuario")["id"]
        .nunique()
        .reset_index(name="productos_actualizados")
    )

    # Merge por usuario
    act_merge = pd.merge(
        crea_user,
        act_user,
        on="usuario",
        how="outer",
    ).fillna(0)

    # Quitamos usuarios sin actividad (0 creados y 0 actualizados)
    act_merge = act_merge[
        (act_merge["productos_creados"] > 0) | (act_merge["productos_actualizados"] > 0)
    ]

    # Ordenamos por creados y luego actualizados, y nos quedamos con el top 20
    act_merge = act_merge.sort_values(
        ["productos_creados", "productos_actualizados"], ascending=False
    ).head(20)

    if act_merge.empty:
        if modo_usuarios == "Solo año seleccionado":
            st.info(f"No hay actividad de usuarios sobre productos en {year}.")
        else:
            st.info("No hay actividad de usuarios sobre productos.")
    else:
        chart_users_crea = (
            alt.Chart(act_merge)
            .mark_bar()
            .encode(
                x=alt.X("productos_creados:Q", title=f"Productos creados{sufijo_titulo}"),
                y=alt.Y("usuario:N", sort="-x", title="Usuario"),
                tooltip=[
                    alt.Tooltip("usuario:N", title="Usuario"),
                    alt.Tooltip("productos_creados:Q", title="Creados (ids únicos)"),
                    alt.Tooltip("productos_actualizados:Q", title="Actualizados (ids únicos)"),
                ],
            )
            .properties(title=f"Top 20 – Productos creados por usuario {sufijo_titulo}", height=400)
        )

        chart_users_act = (
            alt.Chart(act_merge)
            .mark_bar()
            .encode(
                x=alt.X("productos_actualizados:Q", title=f"Productos actualizados{sufijo_titulo}"),
                y=alt.Y("usuario:N", sort="-x", title="Usuario"),
                tooltip=[
                    alt.Tooltip("usuario:N", title="Usuario"),
                    alt.Tooltip("productos_creados:Q", title="Creados (ids únicos)"),
                    alt.Tooltip("productos_actualizados:Q", title="Actualizados (ids únicos)"),
                ],
            )
            .properties(title=f"Top 20 – Productos actualizados por usuario {sufijo_titulo}", height=400)
        )

        st.altair_chart(chart_users_crea | chart_users_act, use_container_width=True)

        st.markdown("### 📄 Detalle por usuario (Top 20)")

        usuarios_lista = ["(Todos)"] + act_merge["usuario"].tolist()
        usuario_sel = st.selectbox("Ver detalle de productos trabajados por:", usuarios_lista)

        if usuario_sel == "(Todos)":
            detalle_df = prod[
                (mask_crea_base | mask_act_base)
                & (
                    prod["usuario_crea_clean"].isin(act_merge["usuario"])
                    | prod["usuario_actualiza_clean"].isin(act_merge["usuario"])
                )
            ][["id", "created_at", "updated_at", "usuario_crea", "usuario_actualiza"]]
        else:
            detalle_df = prod[
                ((mask_crea_base & (prod["usuario_crea_clean"] == usuario_sel)) |
                 (mask_act_base & (prod["usuario_actualiza_clean"] == usuario_sel)))
            ][["id", "created_at", "updated_at", "usuario_crea", "usuario_actualiza"]]

        st.dataframe(detalle_df.sort_values(["id", "created_at"]))
