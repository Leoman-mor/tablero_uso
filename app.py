# app.py
# -----------------------------------------------------------
# Dashboard de Usabilidad Soluquim
#   - empresas.csv: uso de la plataforma
#   - productos.csv: creación / actualización de productos
#   - 5 pestañas:
#       1) Uso en el tiempo y profundidad
#       2) Conversión y flujos
#       3) Ranking de empresas (índice de usabilidad)
#       4) Tiempos hasta descargas (FDS / etiquetas)
#       5) Productos
# -----------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime

# -----------------------------------------------------------
# CONFIG GENERAL
# -----------------------------------------------------------
st.set_page_config(
    page_title="Usabilidad Soluquim",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
    <style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
        max-width: 1400px;
    }
    .big-metric {
        font-size: 26px;
        font-weight: 700;
        margin-bottom: -6px;
        color: #004c6d;
    }
    .metric-label {
        color: #555;
        font-size: 13px;
    }
    .small-help {
        color: #777;
        font-size: 12px;
    }
    .metric-card {
        padding: 0.75rem 0.6rem;
        border-radius: 0.75rem;
        border: 1px solid #e0e0e0;
        background-color: #fafafa;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# CARGA DE DATOS (CON CACHE)
# -----------------------------------------------------------
@st.cache_data
def load_empresas():
    df = pd.read_csv("empresas.csv")
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["fecha"] = df["created_at"].dt.date
    df = df.sort_values(["empresa", "usuario", "created_at"])
    return df

@st.cache_data
def load_productos():
    dfp = pd.read_csv("productos.csv")

    # Parseo de fechas (incluye la nueva fecha_adjunto)
    for col in ["created_at", "updated_at", "fecha_adjunto"]:
        if col in dfp.columns:
            dfp[col] = pd.to_datetime(dfp[col], errors="coerce")

    if "Unnamed: 0" in dfp.columns:
        dfp = dfp.drop(columns=["Unnamed: 0"])
    return dfp


# -----------------------------------------------------------
# BOTÓN PARA ACTUALIZAR DATOS (LIMPIAR CACHE)
# -----------------------------------------------------------
col_refresh, _ = st.columns([1, 5])
with col_refresh:
    if st.button("🔄 Actualizar datos"):
        load_empresas.clear()
        load_productos.clear()
        st.success("Datos recargados desde los CSV.")

# Después del botón, SIEMPRE cargamos desde cache (o desde cero si se limpió)
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

def seleccionar_rango_fechas(df: pd.DataFrame, key_prefix: str):
    """
    Selector de rango de fechas con 3 modos:
    - Años (rango de años)
    - Meses (rango de año-mes)
    - Fechas exactas (date_input)

    Maneja también casos con 0 o 1 año/mes para evitar errores de rango.
    """
    min_fecha = df["fecha"].min()
    max_fecha = df["fecha"].max()

    st.markdown("**Periodo de análisis**")
    col_modo, col_widget = st.columns([1, 3])

    with col_modo:
        modo = st.radio(
            "Modo de selección de fechas",
            ["Años", "Meses", "Fechas exactas"],
            horizontal=True,
            key=f"{key_prefix}_modo_fecha",
            label_visibility="collapsed",
        )

    # ---------------- MODO AÑOS ----------------
    if modo == "Años":
        years = sorted(df["created_at"].dt.year.dropna().unique().tolist())

        # Sin años detectados → usamos todo el rango disponible
        if len(years) == 0:
            with col_widget:
                st.caption("No se detectaron años en los datos; se usa el rango completo.")
            fecha_ini, fecha_fin = min_fecha, max_fecha

        # Un solo año → no usamos slider de rango
        elif len(years) == 1:
            year = years[0]
            with col_widget:
                st.markdown(f"Año disponible: **{year}**")
            fecha_ini = datetime(year, 1, 1).date()
            fecha_fin = datetime(year, 12, 31).date()

        # Varios años → slider de rango normal
        else:
            with col_widget:
                año_ini, año_fin = st.select_slider(
                    "Rango de años",
                    options=years,
                    value=(years[0], years[-1]),
                    key=f"{key_prefix}_rango_anios",
                )
            fecha_ini = datetime(año_ini, 1, 1).date()
            fecha_fin = datetime(año_fin, 12, 31).date()

    # ---------------- MODO MESES ----------------
    elif modo == "Meses":
        periodos = df["created_at"].dt.to_period("M").dropna().unique()
        periodos = sorted(periodos)
        labels = [str(p) for p in periodos]

        # Sin meses → rango completo
        if len(labels) == 0:
            with col_widget:
                st.caption("No se detectaron meses en los datos; se usa el rango completo.")
            fecha_ini, fecha_fin = min_fecha, max_fecha

        # Un solo mes → sin slider de rango
        elif len(labels) == 1:
            unico = labels[0]
            with col_widget:
                st.markdown(f"Mes disponible: **{unico}**")
            p = pd.Period(unico)
            fecha_ini = p.start_time.date()
            fecha_fin = p.end_time.date()

        # Varios meses → slider de rango normal
        else:
            with col_widget:
                val_ini, val_fin = st.select_slider(
                    "Rango de meses (AAAA-MM)",
                    options=labels,
                    value=(labels[0], labels[-1]),
                    key=f"{key_prefix}_rango_meses",
                )
            p_ini = pd.Period(val_ini)
            p_fin = pd.Period(val_fin)
            fecha_ini = p_ini.start_time.date()
            fecha_fin = p_fin.end_time.date()

    # ---------------- FECHAS EXACTAS ----------------
    else:  # Fechas exactas
        with col_widget:
            rango = st.date_input(
                "Rango de fechas",
                value=(min_fecha, max_fecha),
                min_value=min_fecha,
                max_value=max_fecha,
                key=f"{key_prefix}_rango_fechas_exacto",
            )

        if isinstance(rango, tuple):
            fecha_ini, fecha_fin = rango
        else:
            fecha_ini, fecha_fin = rango, rango

    return fecha_ini, fecha_fin


def seleccionar_config_sesion(key_prefix: str):
    """
    Opciones de sesión ocultas en un expander.
    Devuelve (modo_sesion, session_timeout).
    """
    with st.expander("⚙️ Opciones de sesión (avanzado)", expanded=False):
        modo_label = st.radio(
            "Definición de sesión",
            [
                "Por inicio de sesión (Ingreso a login)",
                "Por inactividad del usuario (X min sin eventos)",
            ],
            key=f"{key_prefix}_modo_sesion",
        )

        if modo_label.startswith("Por inicio"):
            modo = "login"
            timeout = 30
            st.caption(
                "Sesión = desde un 'Ingreso a login' hasta el siguiente, para ese usuario en esa empresa."
            )
        else:
            modo = "inactividad"
            timeout = st.slider(
                "Umbral de inactividad por usuario (minutos)",
                min_value=5,
                max_value=90,
                value=30,
                step=5,
                key=f"{key_prefix}_timeout",
            )
            st.caption(
                "Sesión = bloque de acciones del mismo usuario sin pausas mayores a este tiempo."
            )

    return modo, timeout

# -----------------------------------------------------------
# TÍTULO Y GLOSARIO
# -----------------------------------------------------------
st.title("📊 Usabilidad de Soluquim")
st.caption(
    "Vista ejecutiva para entender cómo las empresas usan la plataforma: frecuencia, profundidad, conversión, "
    "tiempos hasta descargas y trabajo sobre productos."
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
- **Tiempos hasta descargas:** minutos promedio que tarda una sesión en llegar a una descarga de FDS o etiqueta desde que entra al módulo de productos.
- **Productos trabajados:** conteo de productos distintos (id) creados o actualizados en un periodo. 
  - Una creación/actualización solo se cuenta una vez por combinación (id, usuario).
- **Productos creados:** productos con fecha de creación en el año seleccionado y usuario de creación definido.
- **Productos actualizados:** productos con al menos una actualización en el año seleccionado y usuario de actualización definido.
- **Actividad por usuario:** número de productos en los que cada usuario aparece como creador o actualizador (deduplicado por producto). Top 20.
        """
    )

# -----------------------------------------------------------
# PESTAÑAS
# -----------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📈 Uso en el tiempo y profundidad",
        "🎯 Conversión y flujos",
        "🏢 Ranking de empresas",
        "⏱️ Tiempos hasta descargas",
        "📦 Productos",
    ]
)

# -----------------------------------------------------------
# TAB 1 – FRECUENCIA Y PROFUNDIDAD
# -----------------------------------------------------------
with tab1:
    st.subheader("Uso en el tiempo y profundidad")

    # 1) Rango de fechas (solo esta pestaña)
    fecha_ini, fecha_fin = seleccionar_rango_fechas(df_raw, "tab1")

    df_tab_base = df_raw[df_raw["fecha"].between(fecha_ini, fecha_fin)].copy()
    if df_tab_base.empty:
        st.info("No hay datos de uso para el rango de fechas seleccionado.")
    else:
        # 2) Configuración de sesiones (oculta)
        modo_sesion, session_timeout = seleccionar_config_sesion("tab1")

        # 3) Construir sesiones
        df_ses = preparar_sesiones(
            df_tab_base,
            modo_sesion=modo_sesion,
            session_timeout=session_timeout,
        )

        # 4) Filtro de empresa local
        empresas_unicas = sorted(df_ses["empresa"].unique())
        opciones_emp_tab = ["Todas las empresas"] + empresas_unicas
        empresa_sel_tab = st.selectbox(
            "Empresa para el análisis en esta pestaña",
            opciones_emp_tab,
            help="Solo afecta los gráficos de esta pestaña.",
        )

        if empresa_sel_tab == "Todas las empresas":
            df_tab = df_ses.copy()
            num_emp_tab = df_tab["empresa"].nunique()
        else:
            df_tab = df_ses[df_ses["empresa"] == empresa_sel_tab].copy()
            num_emp_tab = 1

        if df_tab.empty:
            st.info("No hay datos de uso para la empresa en el rango seleccionado.")
        else:
            usuarios_activos = df_tab[["empresa", "usuario"]].drop_duplicates().shape[0]
            acciones_por_sesion = df_tab.groupby("session_id")["function_used"].count()

            sesiones_por_dia = (
                df_tab.groupby("fecha")["session_id"]
                .nunique()
                .rename("Sesiones")
                .to_frame()
            )
            sesiones_por_dia.index = pd.to_datetime(sesiones_por_dia.index)

            sesiones_tab = df_tab.groupby("session_id").agg(
                vio_producto=("es_ver", "max"),
                descargo_fds=("es_fds", "max"),
                descargo_etq=("es_etq", "max"),
            )
            sesiones_tab["descargo_algo"] = sesiones_tab["descargo_fds"] | sesiones_tab["descargo_etq"]
            sesiones_con_ver = sesiones_tab[sesiones_tab["vio_producto"]]

            conv_cualquier = (
                sesiones_con_ver["descargo_algo"].mean() if not sesiones_con_ver.empty else 0.0
            )

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.markdown(
                    f"<div class='metric-card'><div class='big-metric'>{num_emp_tab}</div>"
                    "<div class='metric-label'>Empresas con actividad</div></div>",
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    f"<div class='metric-card'><div class='big-metric'>{usuarios_activos}</div>"
                    "<div class='metric-label'>Usuarios activos</div></div>",
                    unsafe_allow_html=True,
                )
            with col3:
                st.markdown(
                    f"<div class='metric-card'><div class='big-metric'>{len(acciones_por_sesion)}</div>"
                    "<div class='metric-label'>Sesiones</div></div>",
                    unsafe_allow_html=True,
                )
            with col4:
                st.markdown(
                    f"<div class='metric-card'><div class='big-metric'>{acciones_por_sesion.mean():.1f}</div>"
                    "<div class='metric-label'>Acciones promedio por sesión</div></div>",
                    unsafe_allow_html=True,
                )
            with col5:
                st.markdown(
                    f"<div class='metric-card'><div class='big-metric'>{conv_cualquier*100:.1f}%</div>"
                    "<div class='metric-label'>Conversión ver → descarga</div></div>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                "<div class='small-help'>Las métricas respetan el rango de fechas, la definición de sesión "
                "y la empresa seleccionada.</div>",
                unsafe_allow_html=True,
            )

            st.markdown("---")
            st.subheader("Frecuencia de uso – sesiones por periodo")

            nivel_tiempo = st.radio(
                "Nivel de detalle del tiempo",
                ["Promedio diario", "Promedio semanal", "Promedio mensual"],
                horizontal=True,
                key="tab1_nivel_tiempo",
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
    st.subheader("Conversión y flujos")

    # 1) Rango de fechas
    fecha_ini, fecha_fin = seleccionar_rango_fechas(df_raw, "tab2")

    df_tab_base = df_raw[df_raw["fecha"].between(fecha_ini, fecha_fin)].copy()
    if df_tab_base.empty:
        st.info("No hay datos de uso para el rango de fechas seleccionado.")
    else:
        # 2) Sesiones
        modo_sesion, session_timeout = seleccionar_config_sesion("tab2")

        df_ses = preparar_sesiones(
            df_tab_base,
            modo_sesion=modo_sesion,
            session_timeout=session_timeout,
        )

        # 3) Empresa local
        empresas_unicas = sorted(df_ses["empresa"].unique())
        opciones_emp_tab = ["Todas las empresas"] + empresas_unicas
        empresa_sel_tab = st.selectbox(
            "Empresa para el análisis en esta pestaña",
            opciones_emp_tab,
            help="Solo afecta los gráficos de esta pestaña.",
            key="tab2_empresa",
        )

        if empresa_sel_tab == "Todas las empresas":
            df_tab = df_ses.copy()
            num_emp_tab = df_tab["empresa"].nunique()
        else:
            df_tab = df_ses[df_ses["empresa"] == empresa_sel_tab].copy()
            num_emp_tab = 1

        if df_tab.empty:
            st.info("No hay datos de uso para la empresa en el rango seleccionado.")
        else:
            usuarios_activos = df_tab[["empresa", "usuario"]].drop_duplicates().shape[0]
            acciones_por_sesion = df_tab.groupby("session_id")["function_used"].count()

            sesiones_conv = df_tab.groupby("session_id").agg(
                vio_producto=("es_ver", "max"),
                descargo_fds=("es_fds", "max"),
                descargo_etq=("es_etq", "max"),
            )
            sesiones_conv["descargo_algo"] = sesiones_conv["descargo_fds"] | sesiones_conv["descargo_etq"]
            sesiones_con_ver = sesiones_conv[sesiones_conv["vio_producto"]]

            conv_fds = sesiones_con_ver["descargo_fds"].mean() if not sesiones_con_ver.empty else 0.0
            conv_etq = sesiones_con_ver["descargo_etq"].mean() if not sesiones_con_ver.empty else 0.0
            conv_cualquier = (
                sesiones_con_ver["descargo_algo"].mean() if not sesiones_con_ver.empty else 0.0
            )

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.markdown(
                    f"<div class='metric-card'><div class='big-metric'>{num_emp_tab}</div>"
                    "<div class='metric-label'>Empresas con actividad</div></div>",
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    f"<div class='metric-card'><div class='big-metric'>{usuarios_activos}</div>"
                    "<div class='metric-label'>Usuarios activos</div></div>",
                    unsafe_allow_html=True,
                )
            with col3:
                st.markdown(
                    f"<div class='metric-card'><div class='big-metric'>{len(acciones_por_sesion)}</div>"
                    "<div class='metric-label'>Sesiones</div></div>",
                    unsafe_allow_html=True,
                )
            with col4:
                st.markdown(
                    f"<div class='metric-card'><div class='big-metric'>{acciones_por_sesion.mean():.1f}</div>"
                    "<div class='metric-label'>Acciones promedio por sesión</div></div>",
                    unsafe_allow_html=True,
                )
            with col5:
                st.markdown(
                    f"<div class='metric-card'><div class='big-metric'>{conv_cualquier*100:.1f}%</div>"
                    "<div class='metric-label'>Conversión ver → descarga</div></div>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                "<div class='small-help'>Las métricas respetan el rango de fechas, la definición de sesión "
                "y la empresa seleccionada.</div>",
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
                "Las tasas se calculan **solo sobre sesiones donde hubo visualización del detalle de un producto**.",
            )

            st.subheader("Flujos más frecuentes (pares de acciones consecutivas)")

            df_sorted = df_tab.sort_values(["session_id", "created_at"])
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
    st.subheader("Ranking de empresas por usabilidad")

    # 1) Rango de fechas
    fecha_ini, fecha_fin = seleccionar_rango_fechas(df_raw, "tab3")

    df_tab_base = df_raw[df_raw["fecha"].between(fecha_ini, fecha_fin)].copy()
    if df_tab_base.empty:
        st.info("No hay datos de uso para el rango de fechas seleccionado.")
    else:
        # 2) Sesiones
        modo_sesion, session_timeout = seleccionar_config_sesion("tab3")

        df_ses = preparar_sesiones(
            df_tab_base,
            modo_sesion=modo_sesion,
            session_timeout=session_timeout,
        )

        # Métricas globales para cabecera (en este rango)
        usuarios_activos = df_ses[["empresa", "usuario"]].drop_duplicates().shape[0]
        acciones_por_sesion = df_ses.groupby("session_id")["function_used"].count()

        sesiones_global = df_ses.groupby("session_id").agg(
            vio_producto=("es_ver", "max"),
            descargo_fds=("es_fds", "max"),
            descargo_etq=("es_etq", "max"),
        )
        sesiones_global["descargo_algo"] = (
            sesiones_global["descargo_fds"] | sesiones_global["descargo_etq"]
        )
        sesiones_con_ver = sesiones_global[sesiones_global["vio_producto"]]

        conv_cualquier = (
            sesiones_con_ver["descargo_algo"].mean()
            if not sesiones_con_ver.empty
            else 0.0
        )

        emp_count = df_ses["empresa"].nunique()

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(
                f"<div class='metric-card'><div class='big-metric'>{emp_count}</div>"
                "<div class='metric-label'>Empresas con actividad</div></div>",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"<div class='metric-card'><div class='big-metric'>{usuarios_activos}</div>"
                "<div class='metric-label'>Usuarios activos</div></div>",
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f"<div class='metric-card'><div class='big-metric'>{len(acciones_por_sesion)}</div>"
                "<div class='metric-label'>Sesiones</div></div>",
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                f"<div class='metric-card'><div class='big-metric'>{acciones_por_sesion.mean():.1f}</div>"
                "<div class='metric-label'>Acciones promedio por sesión</div></div>",
                unsafe_allow_html=True,
            )
        with col5:
            st.markdown(
                f"<div class='metric-card'><div class='big-metric'>{conv_cualquier*100:.1f}%</div>"
                "<div class='metric-label'>Conversión ver → descarga</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            "<div class='small-help'>El ranking se construye con las empresas que han tenido uso en el rango de fechas seleccionado.</div>",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.subheader("Ranking por índice de usabilidad")

        # 3) Construir emp (por empresa) en este rango
        sesiones_emp = df_ses.groupby(["empresa", "usuario", "session_id"]).agg(
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

        SESIONES_CONFIABLES = 10
        emp["factor_volumen"] = (
            emp["sesiones"].clip(upper=SESIONES_CONFIABLES) / SESIONES_CONFIABLES
        )

        emp["indice_usabilidad"] = emp["indice_base"] * emp["factor_volumen"]

        SESIONES_MINIMAS = 3
        emp_filtrado = emp[emp["sesiones"] >= SESIONES_MINIMAS].copy()

        if emp_filtrado.empty:
            st.info("No hay empresas que cumplan el mínimo de sesiones (≥ 3).")
        else:
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
                key="tab3_metrica",
            )
            metrica_col = opciones_ranking[metrica_label]

            emp_ordenado = (
                emp_filtrado.sort_values(metrica_col, ascending=False)
                .reset_index()
            )
            emp_ordenado["Puesto"] = emp_ordenado.index + 1

            n_emp = emp_ordenado.shape[0]

            if n_emp == 1:
                top_n = 1
                st.caption("Solo hay 1 empresa en el ranking; se muestra completa.")
            else:
                max_empresas = min(30, n_emp)
                top_n = st.slider(
                    "Número de empresas a mostrar en la gráfica",
                    min_value=1,
                    max_value=max_empresas,
                    value=min(15, max_empresas),
                    key="tab3_top_n",
                )

            emp_top = emp_ordenado.head(top_n)

            rank_chart = (
                alt.Chart(emp_top)
                .mark_bar()
                .encode(
                    y=alt.Y("empresa:N", sort="-x", title="Empresa"),
                    x=alt.X(f"{metrica_col}:Q", title=metrica_label),
                    tooltip=[
                        alt.Tooltip("Puesto:Q", title="Puesto"),
                        alt.Tooltip("empresa:N", title="Empresa"),
                        alt.Tooltip("sesiones:Q", title="Sesiones"),
                        alt.Tooltip("eventos_tot:Q", title="Acciones totales"),
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

            # -------- FICHA RÁPIDA POR EMPRESA --------
            st.markdown("### Ficha rápida por empresa")

            # Todas las empresas con actividad en el periodo, ordenadas ABC sin importar mayúsculas
            empresas_abc = sorted(
                df_ses["empresa"].dropna().unique().tolist(),
                key=lambda x: x.upper()
            )


            empresa_focus = st.selectbox(
                "Selecciona una empresa para ver su puesto y métricas",
                empresas_abc,
                key="tab3_select_empresa",
            )

            if empresa_focus in emp_ordenado["empresa"].values:
                fila = emp_ordenado[emp_ordenado["empresa"] == empresa_focus].iloc[0]
                st.markdown(
                    f"""
**{empresa_focus} – Ficha de usabilidad**

- Puesto en el ranking ({metrica_label}): **#{fila['Puesto']}** de {n_emp}
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
            else:
                st.warning(
                    f"La empresa **{empresa_focus}** no aparece en el ranking porque no cumple "
                    f"el mínimo de sesiones ({SESIONES_MINIMAS})."
                )

            # -------- TABLA COMPLETA EN EXPANDER --------
            st.markdown("### 📄 Detalle completo del ranking")

            tabla_ranking = emp_ordenado.rename(
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
                    "Puesto",
                    "Empresa",
                    "Sesiones",
                    "Acciones totales",
                    "Acciones prom. por sesión",
                    "Conv. promedio (FDS+etq)",
                    "Índice base (0–1)",
                    "Índice de usabilidad (0–1)",
                ]
            ]

            with st.expander("Ver tabla de ranking completa", expanded=False):
                st.dataframe(tabla_ranking)


# -----------------------------------------------------------
# TAB 4 – TIEMPOS HASTA DESCARGAS
# -----------------------------------------------------------
with tab4:
    st.subheader("⏱️ Tiempos para llegar a descargas de FDS / etiquetas")

    st.markdown(
        "<div class='small-help'>"
        "Medimos cuánto tardan las sesiones, en promedio, en generar una descarga de FDS o de etiquetas "
        "desde que los usuarios entran al módulo de productos."
        "</div>",
        unsafe_allow_html=True,
    )

    # 1) Rango de fechas
    fecha_ini, fecha_fin = seleccionar_rango_fechas(df_raw, "tab4")

    df_tab_base = df_raw[df_raw["fecha"].between(fecha_ini, fecha_fin)].copy()
    if df_tab_base.empty:
        st.info("No hay datos de uso para el rango de fechas seleccionado.")
    else:
        # 2) Sesiones
        modo_sesion, session_timeout = seleccionar_config_sesion("tab4")

        df_ses = preparar_sesiones(
            df_tab_base,
            modo_sesion=modo_sesion,
            session_timeout=session_timeout,
        )

        df_ses = df_ses.sort_values(["session_id", "created_at"]).copy()

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

        mod_times = (
            df_ses[df_ses["function_used"] == "Acceso al módulo de productos"]
            .groupby("session_id")["created_at"]
            .min()
            .rename("t_modulo_productos")
        )

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

        ver_times = (
            df_ses[df_ses["es_ver"]]
            .groupby("session_id")["created_at"]
            .min()
            .rename("t_ver")
        )

        ses = (
            ses_base
            .merge(mod_times, on="session_id", how="left")
            .merge(ver_times, on="session_id", how="left")
            .merge(fds_times, on="session_id", how="left")
            .merge(etq_times, on="session_id", how="left")
        )

        ses["t_inicio_trabajo"] = (
            ses["t_modulo_productos"]
            .combine_first(ses["t_ver"])
            .combine_first(ses["inicio_sesion"])
        )

        def diff_minutos(col_fin: str) -> pd.Series:
            return (
                (ses[col_fin] - ses["t_inicio_trabajo"])
                .dt.total_seconds()
                .div(60)
            )

        ses["min_hasta_fds"] = diff_minutos("t_fds")
        ses["min_hasta_etq"] = diff_minutos("t_etq")

        ses["t_desc_cualquiera"] = ses[["t_fds", "t_etq"]].min(axis=1)
        ses["min_hasta_desc_cualquiera"] = diff_minutos("t_desc_cualquiera")

        total_sesiones = ses["session_id"].nunique()
        if total_sesiones == 0:
            st.info("No hay sesiones en el rango seleccionado.")
        else:
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
                    "en el rango seleccionado."
                )
            else:
                sesiones_con_fds = ses.loc[mask_fds, "session_id"].nunique()
                sesiones_con_etq = ses.loc[mask_etq, "session_id"].nunique()
                sesiones_con_desc_any = ses.loc[mask_any, "session_id"].nunique()
                empresas_con_desc = ses.loc[mask_any, "empresa"].nunique()

                diffs_fds = ses.loc[mask_fds, "min_hasta_fds"]
                diffs_etq = ses.loc[mask_etq, "min_hasta_etq"]
                diffs_any = ses.loc[mask_any, "min_hasta_desc_cualquiera"]

                prom_fds = diffs_fds.mean() if not diffs_fds.empty else np.nan
                prom_etq = diffs_etq.mean() if not diffs_etq.empty else np.nan
                prom_any = diffs_any.mean() if not diffs_any.empty else np.nan  # por si lo usas luego

                col_a, col_b, col_c, col_d = st.columns(4)

                col_a.markdown(
                    f"<div class='metric-card'><div class='big-metric'>{empresas_con_desc}</div>"
                    "<div class='metric-label'>Empresas con descargas</div></div>",
                    unsafe_allow_html=True,
                )
                col_b.markdown(
                    f"<div class='metric-card'><div class='big-metric'>{sesiones_con_desc_any}</div>"
                    f"<div class='metric-label'>Sesiones con descargas ({sesiones_con_desc_any/total_sesiones*100:.1f}% del total)</div></div>",
                    unsafe_allow_html=True,
                )

                if not np.isnan(prom_fds):
                    col_c.markdown(
                        f"<div class='metric-card'><div class='big-metric'>{prom_fds:.1f}</div>"
                        "<div class='metric-label'>Minutos prom. hasta 1ª FDS</div></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    col_c.markdown(
                        "<div class='metric-card'><div class='big-metric'>–</div>"
                        "<div class='metric-label'>Minutos prom. hasta 1ª FDS</div></div>",
                        unsafe_allow_html=True,
                    )

                if not np.isnan(prom_etq):
                    col_d.markdown(
                        f"<div class='metric-card'><div class='big-metric'>{prom_etq:.1f}</div>"
                        "<div class='metric-label'>Minutos prom. hasta 1ª etiqueta</div></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    col_d.markdown(
                        "<div class='metric-card'><div class='big-metric'>–</div>"
                        "<div class='metric-label'>Minutos prom. hasta 1ª etiqueta</div></div>",
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    "<div class='small-help'>"
                    "Solo se consideran tiempos entre 0 y 120 minutos para evitar casos extremos."
                    "</div>",
                    unsafe_allow_html=True,
                )

                st.markdown("---")
                st.subheader("🏢 Ranking de empresas por tiempo promedio")

                ses_tot_emp = (
                    ses.groupby("empresa")["session_id"]
                    .nunique()
                    .reset_index(name="sesiones_totales")
                )

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

                ranking = (
                    ses_tot_emp
                    .merge(fds_emp, on="empresa", how="left")
                    .merge(etq_emp, on="empresa", how="left")
                )

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
                    orden_opcion = st.radio(
                        "Orden del ranking",
                        ["Más rápidos primero", "Más lentos primero"],
                        horizontal=True,
                        key="tab4_orden",
                        help="Elige si quieres ver primero las empresas más ágiles o las más lentas.",
                    )
                    asc = True if orden_opcion == "Más rápidos primero" else False
                    sort_order = "ascending" if asc else "descending"

                    if num_emp_disp == 1:
                        top_n = 1
                        st.caption(
                            "Solo hay 1 empresa con suficientes descargas; se muestra completa."
                        )
                    else:
                        max_empresas = min(30, num_emp_disp)
                        top_n = st.slider(
                            "Número de empresas a mostrar en el ranking",
                            min_value=1,
                            max_value=max_empresas,
                            value=min(15, max_empresas),
                            key="tab4_top_n",
                            help="Controla cuántas empresas aparecen en las barras.",
                        )

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
                                        title="Minutos prom. hasta 1ª FDS",
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
                                        title="Minutos prom. hasta 1ª etiqueta",
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

                sort_col_candidates = [
                    "Tiempo prom. hasta 1ª FDS (min)",
                    "Tiempo prom. hasta 1ª etiqueta (min)",
                ]
                sort_col = next((c for c in sort_col_candidates if c in tabla.columns), None)

                if sort_col:
                    # usa el mismo sentido de orden que el ranking (si existe asc)
                    asc_tabla = True if 'asc' in locals() and asc else False
                    tabla_sorted = tabla[cols_tabla].sort_values(
                        sort_col,
                        ascending=asc_tabla,
                        na_position="last",
                    )
                else:
                    tabla_sorted = tabla[cols_tabla]

                st.dataframe(tabla_sorted)

                st.markdown("### 🔍 Ficha de tiempos por empresa")

                empresas_disponibles = ranking["empresa"].unique().tolist()
                empresa_focus = st.selectbox(
                    "Selecciona una empresa para ver sus tiempos",
                    sorted(empresas_disponibles, key=lambda x: x.upper()),
                    key="tab4_empresa_focus",
                )

                if empresa_focus:
                    fila = ranking[ranking["empresa"] == empresa_focus].iloc[0]
                    st.markdown(
                        f"""
**{empresa_focus} – Tiempos de descarga**

- Sesiones totales en el periodo: **{fila['sesiones_totales']}**
- Sesiones con FDS: **{fila.get('sesiones_con_fds', 0)}**
- Sesiones con etiquetas: **{fila.get('sesiones_con_etq', 0)}**
- Tiempo promedio hasta 1ª FDS (min): **{fila.get('tiempo_prom_fds', float('nan')):.2f}**
- Tiempo mediano hasta 1ª FDS (min): **{fila.get('tiempo_mediano_fds', float('nan')):.2f}**
- Tiempo promedio hasta 1ª etiqueta (min): **{fila.get('tiempo_prom_etq', float('nan')):.2f}**
- Tiempo mediano hasta 1ª etiqueta (min): **{fila.get('tiempo_mediano_etq', float('nan')):.2f}**
                        """
                    )

# -----------------------------------------------------------
# TAB 5 – PRODUCTOS (CREACIÓN / ACTUALIZACIÓN)
# -----------------------------------------------------------
with tab5:
    st.subheader("📦 Productos creados y actualizados")

    prod = df_prod_raw.copy()

    # Aseguramos tipos de fecha por si viene algo raro del CSV
    prod["created_at"] = pd.to_datetime(prod["created_at"], errors="coerce")
    prod["updated_at"] = pd.to_datetime(prod["updated_at"], errors="coerce")
    if "fecha_adjunto" in prod.columns:
        prod["fecha_adjunto"] = pd.to_datetime(prod["fecha_adjunto"], errors="coerce")

    # Limpieza de usuarios
    prod["usuario_crea_clean"] = prod["usuario_crea"].fillna("").astype(str).str.strip()
    prod["usuario_actualiza_clean"] = prod["usuario_actualiza"].fillna("").astype(str).str.strip()

    # Años disponibles (creación / actualización)
    years_created = prod["created_at"].dt.year.dropna().astype(int)
    years_updated = prod["updated_at"].dt.year.dropna().astype(int)
    years_all = sorted(set(years_created.tolist()) | set(years_updated.tolist()))

    if not years_all:
        st.info("No hay información de fechas de creación/actualización de productos.")
    else:
        # ---------------- Selección de año y métricas principales ----------------
        default_year = 2025 if 2025 in years_all else max(years_all)
        year = st.selectbox(
            "Año a analizar",
            years_all,
            index=years_all.index(default_year),
        )

        mask_created_year = prod["created_at"].dt.year == year
        mask_updated_year = prod["updated_at"].dt.year == year

        mask_crea_user_valido = prod["usuario_crea_clean"] != ""
        mask_act_user_valido = prod["usuario_actualiza_clean"] != ""

        total_productos = prod["id"].nunique()

        creados_ano = prod.loc[
            mask_created_year & mask_crea_user_valido, "id"
        ].nunique()

        actualizados_ano = prod.loc[
            mask_updated_year & mask_act_user_valido, "id"
        ].nunique()

        c1, c2, c3 = st.columns(3)
        c1.markdown(
            f"<div class='metric-card'><div class='big-metric'>{total_productos}</div>"
            "<div class='metric-label'>Productos totales en Soluquim</div></div>",
            unsafe_allow_html=True,
        )
        c2.markdown(
            f"<div class='metric-card'><div class='big-metric'>{creados_ano}</div>"
            f"<div class='metric-label'>Productos creados en {year}</div></div>",
            unsafe_allow_html=True,
        )
        c3.markdown(
            f"<div class='metric-card'><div class='big-metric'>{actualizados_ano}</div>"
            f"<div class='metric-label'>Productos actualizados en {year}</div></div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div class='small-help'>El total incluye todos los ids; las métricas por año se basan en created_at/updated_at "
            "de ese año, solo con usuarios definidos.</div>",
            unsafe_allow_html=True,
        )

        # ---------------- Análisis de adjuntos (FDS) ----------------
        st.markdown("### 📎 Análisis de adjuntos (FDS)")

        if "fecha_adjunto" not in prod.columns:
            st.info("Este dataset aún no trae la columna 'fecha_adjunto'.")
        else:
            # Nos quedamos solo con filas con fecha_adjunto válida
            prod_adj = prod[prod["fecha_adjunto"].notna()].copy()

            if prod_adj.empty:
                st.info("No hay productos con adjuntos registrados.")
            else:

                # Año y mes del adjunto
                prod_adj["anio"] = prod_adj["fecha_adjunto"].dt.year.astype(int)
                prod_adj["mes_num"] = prod_adj["fecha_adjunto"].dt.month

                mapa_meses = {
                    1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr",
                    5: "May", 6: "Jun", 7: "Jul", 8: "Ago",
                    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic",
                }
                prod_adj["mes_label"] = prod_adj["mes_num"].map(mapa_meses)

                # Antigüedad de las FDS (regla de 5 años)
                hoy = pd.Timestamp.today().normalize()
                prod_adj["antig_dias"] = (hoy - prod_adj["fecha_adjunto"]).dt.days
                prod_adj["antig_anios"] = prod_adj["antig_dias"] / 365.25

                # --------- MÉTRICAS PARA TARJETAS ----------
                productos_con_adjuntos_total = prod_adj["id"].nunique()
                productos_vencidos_5 = prod_adj.loc[
                    prod_adj["antig_anios"] > 5, "id"
                ].nunique()

                # Fecha promedio de los adjuntos
                fecha_prom_ts = prod_adj["fecha_adjunto"].mean()
                if pd.isna(fecha_prom_ts):
                    fecha_promedio_str = "–"
                else:
                    fecha_promedio_str = fecha_prom_ts.date().isoformat()

                # Para el texto de rango (lo dejamos porque sigue siendo útil)
                fecha_mas_antigua = prod_adj["fecha_adjunto"].min().date()
                fecha_mas_reciente = prod_adj["fecha_adjunto"].max().date()

                col_a, col_b, col_c = st.columns(3)
                col_a.markdown(
                    f"<div class='metric-card'><div class='big-metric'>{productos_con_adjuntos_total}</div>"
                    "<div class='metric-label'>Productos con adjunto</div></div>",
                    unsafe_allow_html=True,
                )
                col_b.markdown(
                    f"<div class='metric-card'><div class='big-metric'>{fecha_promedio_str}</div>"
                    "<div class='metric-label'>Fecha promedio de adjuntos</div></div>",
                    unsafe_allow_html=True,
                )
                col_c.markdown(
                    f"<div class='metric-card'><div class='big-metric'>{productos_vencidos_5}</div>"
                    "<div class='metric-label'>Productos con FDS &gt; 5 años</div></div>",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"<div class='small-help'>Rango de fechas de adjuntos registrados: "
                    f"de <b>{fecha_mas_antigua}</b> a <b>{fecha_mas_reciente}</b>. "
                    "La antigüedad se calcula frente a la fecha actual, usando un umbral de 5 años.</div>",
                    unsafe_allow_html=True,
                )

                st.markdown("---")
                st.markdown("#### 📅 Vista temporal de adjuntos")

                vista_adj = st.radio(
                    "Tipo de análisis",
                    ["Por año", "Por año y mes (heatmap)"],
                    horizontal=True,
                    key="vista_adjuntos",
                )

                # ---- Serie anual: cuántos productos tienen adjunto por año ----
                serie_anual = (
                    prod_adj.groupby("anio")["id"]
                    .nunique()
                    .reset_index(name="productos_con_adjuntos")
                    .sort_values("anio")
                )

                chart_anual = (
                    alt.Chart(serie_anual)
                    .mark_bar()
                    .encode(
                        x=alt.X(
                            "anio:O",
                            title="Año de adjunto",
                            axis=alt.Axis(
                                labelAngle=-45,      # 🔄 rotamos etiquetas
                                labelOverlap=False,  # ❌ no permitir solapamiento
                            ),
                        ),
                        y=alt.Y(
                            "productos_con_adjuntos:Q",
                            title="Productos con adjuntos (ids únicos)",
                        ),
                        tooltip=[
                            alt.Tooltip("anio:O", title="Año"),
                            alt.Tooltip(
                                "productos_con_adjuntos:Q",
                                title="Productos con adjuntos",
                            ),
                        ],
                    )
                    .properties(
                        height=320, 
                    )
                )

                # ---- Heatmap año × mes ----
                serie_heat = (
                    prod_adj.groupby(["anio", "mes_label"])["id"]
                    .nunique()
                    .reset_index(name="productos")
                )

                # Orden correcto de los meses
                serie_heat["mes_label"] = pd.Categorical(
                    serie_heat["mes_label"],
                    categories=[
                        "Ene", "Feb", "Mar", "Abr", "May", "Jun",
                        "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"
                    ],
                    ordered=True,
                )

                heat_chart = (
                    alt.Chart(serie_heat)
                    .mark_rect()
                    .encode(
                        x=alt.X(
                            "mes_label:N",
                            title="Mes",
                            sort=[
                                "Ene", "Feb", "Mar", "Abr", "May", "Jun",
                                "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"
                            ],
                        ),
                        y=alt.Y(
                            "anio:O",
                            title="Año",
                            sort="ascending",
                        ),
                        color=alt.Color(
                            "productos:Q",
                            title="Productos con adjuntos",
                        ),
                        tooltip=[
                            alt.Tooltip("anio:O", title="Año"),
                            alt.Tooltip("mes_label:N", title="Mes"),
                            alt.Tooltip("productos:Q", title="Productos con adjuntos"),
                        ],
                    )
                    .properties(
                        height=320,
                    )
                )

                if vista_adj == "Por año":
                    st.altair_chart(chart_anual, use_container_width=True)
                else:
                    st.altair_chart(heat_chart, use_container_width=True)

                with st.expander("📄 Ver tabla resumida de adjuntos por año y mes"):
                    tabla_adj = (
                        serie_heat.sort_values(["anio", "mes_label"])
                        .rename(
                            columns={
                                "anio": "Año",
                                "mes_label": "Mes",
                                "productos": "Productos con adjuntos",
                            }
                        )
                        .reset_index(drop=True)
                    )
                    st.dataframe(tabla_adj, use_container_width=True)


                # ---------------- Serie mensual de creación / actualización ----------------
        st.markdown("### 📈 Creación y actualización mensual de productos")

        vista_serie = st.radio(
            "Periodo para la serie",
            ["Solo año seleccionado", "Histórico completo"],
            horizontal=True,
            key="tab5_vista_serie",
        )

        # ---------------------------------------------------
        # 1) SOLO AÑO SELECCIONADO → LÍNEA (igual que antes)
        # ---------------------------------------------------
        if vista_serie == "Solo año seleccionado":
            prod_year_created = prod.loc[
                mask_created_year & mask_crea_user_valido
            ].copy()
            prod_year_updated = prod.loc[
                mask_updated_year & mask_act_user_valido
            ].copy()

            if not prod_year_created.empty:
                prod_year_created["mes"] = (
                    prod_year_created["created_at"].dt.to_period("M").astype(str)
                )
            if not prod_year_updated.empty:
                prod_year_updated["mes"] = (
                    prod_year_updated["updated_at"].dt.to_period("M").astype(str)
                )

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

                serie["mes_num"] = pd.to_datetime(serie["mes"] + "-01").dt.month
                mapa_meses = {
                    1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr",
                    5: "May", 6: "Jun", 7: "Jul", 8: "Ago",
                    9: "Sep", 10: "Oct", 11: "Dic",
                }
                # Corrijo mapa_meses (faltaba Nov):
                mapa_meses[11] = "Nov"

                serie["mes_label"] = serie["mes_num"].map(mapa_meses)

                serie_long = serie.melt(
                    id_vars=["mes", "mes_num", "mes_label"],
                    value_vars=["creados", "actualizados"],
                    var_name="tipo",
                    value_name="productos",
                )

                serie_long["tipo"] = serie_long["tipo"].replace(
                    {"creados": "Creados", "actualizados": "Actualizados"}
                )

                chart_line = (
                    alt.Chart(serie_long)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X(
                            "mes_label:N",
                            title="Mes",
                            sort=["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                                  "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"],
                        ),
                        y=alt.Y("productos:Q", title="Productos trabajados"),
                        color=alt.Color("tipo:N", title="Tipo"),
                        tooltip=[
                            alt.Tooltip("mes_label:N", title="Mes"),
                            alt.Tooltip("tipo:N", title="Tipo"),
                            alt.Tooltip("productos:Q", title="Productos"),
                        ],
                    )
                    .properties(
                        title=f"Productos creados y actualizados por mes en {year}",
                        height=320,
                    )
                )

                st.altair_chart(chart_line, use_container_width=True)
            else:
                st.info(f"No hay productos trabajados (con responsable) en {year}.")

                # ---------------------------------------------------
        # 2) HISTÓRICO COMPLETO → DOS GRÁFICAS (CREADOS / ACTUALIZADOS)
        # ---------------------------------------------------
        else:
            # Histórico: usamos todos los años donde haya usuario válido
            crea_hist = prod.loc[mask_crea_user_valido, ["id", "created_at"]].copy()
            act_hist = prod.loc[mask_act_user_valido, ["id", "updated_at"]].copy()

            if crea_hist.empty and act_hist.empty:
                st.info("No hay productos trabajados en el histórico.")
            else:
                crea_hist["anio"] = crea_hist["created_at"].dt.year
                crea_hist["mes_num"] = crea_hist["created_at"].dt.month

                act_hist["anio"] = act_hist["updated_at"].dt.year
                act_hist["mes_num"] = act_hist["updated_at"].dt.month

                # Agrupamos por año y mes (productos únicos)
                serie_crea_hist = (
                    crea_hist
                    .dropna(subset=["anio", "mes_num"])
                    .drop_duplicates(subset=["id", "anio", "mes_num"])
                    .groupby(["anio", "mes_num"])["id"]
                    .nunique()
                    .reset_index(name="creados")
                    if not crea_hist.empty
                    else pd.DataFrame(columns=["anio", "mes_num", "creados"])
                )

                serie_act_hist = (
                    act_hist
                    .dropna(subset=["anio", "mes_num"])
                    .drop_duplicates(subset=["id", "anio", "mes_num"])
                    .groupby(["anio", "mes_num"])["id"]
                    .nunique()
                    .reset_index(name="actualizados")
                    if not act_hist.empty
                    else pd.DataFrame(columns=["anio", "mes_num", "actualizados"])
                )

                serie_hist = pd.merge(
                    serie_crea_hist,
                    serie_act_hist,
                    on=["anio", "mes_num"],
                    how="outer",
                ).fillna(0)

                if serie_hist.empty:
                    st.info("No hay productos trabajados en el histórico.")
                else:
                    serie_hist["total_productos"] = (
                        serie_hist["creados"] + serie_hist["actualizados"]
                    )

                    mapa_meses = {
                        1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr",
                        5: "May", 6: "Jun", 7: "Jul", 8: "Ago",
                        9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic",
                    }
                    serie_hist["mes_label"] = serie_hist["mes_num"].map(mapa_meses)

                    # --- DOS GRÁFICAS LADO A LADO ---
                    col_crea, col_act = st.columns(2)

                    with col_crea:
                        st.markdown("#### 📊 Histórica de productos creados")
                        chart_crea = (
                            alt.Chart(serie_hist)
                            .mark_bar()
                            .encode(
                                x=alt.X(
                                    "mes_label:N",
                                    title="Mes",
                                    sort=[
                                        "Ene", "Feb", "Mar", "Abr", "May", "Jun",
                                        "Jul", "Ago", "Sep", "Oct", "Nov", "Dic",
                                    ],
                                ),
                                xOffset="anio:N",  # una barrita por año en cada mes
                                y=alt.Y(
                                    "creados:Q",
                                    title="Productos creados",
                                ),
                                color=alt.Color(
                                    "anio:N",
                                    title="Año",
                                ),
                                tooltip=[
                                    alt.Tooltip("anio:N", title="Año"),
                                    alt.Tooltip("mes_label:N", title="Mes"),
                                    alt.Tooltip("creados:Q", title="Creados"),
                                ],
                            )
                            .properties(
                                height=320,
                            )
                        )
                        st.altair_chart(chart_crea, use_container_width=True)

                    with col_act:
                        st.markdown("#### 🔁 Histórica de productos actualizados")
                        chart_act = (
                            alt.Chart(serie_hist)
                            .mark_bar()
                            .encode(
                                x=alt.X(
                                    "mes_label:N",
                                    title="Mes",
                                    sort=[
                                        "Ene", "Feb", "Mar", "Abr", "May", "Jun",
                                        "Jul", "Ago", "Sep", "Oct", "Nov", "Dic",
                                    ],
                                ),
                                xOffset="anio:N",
                                y=alt.Y(
                                    "actualizados:Q",
                                    title="Productos actualizados",
                                ),
                                color=alt.Color(
                                    "anio:N",
                                    title="Año",
                                ),
                                tooltip=[
                                    alt.Tooltip("anio:N", title="Año"),
                                    alt.Tooltip("mes_label:N", title="Mes"),
                                    alt.Tooltip("actualizados:Q", title="Actualizados"),
                                ],
                            )
                            .properties(
                                height=320,
                            )
                        )
                        st.altair_chart(chart_act, use_container_width=True)

                    # --- TABLA RESUMEN POR AÑO ---
                    st.markdown("### 📄 Resumen histórico por año")

                    resumen_year = (
                        serie_hist
                        .groupby("anio")
                        .agg(
                            productos_creados=("creados", "sum"),
                            productos_actualizados=("actualizados", "sum"),
                            total_productos=("total_productos", "sum"),
                            meses_con_movimiento=("mes_num", "nunique"),
                        )
                        .reset_index()
                        .sort_values("anio")
                    )

                    resumen_year = resumen_year.rename(
                        columns={
                            "anio": "Año",
                            "productos_creados": "Productos creados",
                            "productos_actualizados": "Productos actualizados",
                            "total_productos": "Total productos",
                            "meses_con_movimiento": "Meses con movimiento",
                        }
                    )

                    st.dataframe(
                        resumen_year.set_index("Año"),
                        use_container_width=True,
                    )




        # ---------------- Actividad por usuario (DINÁMICA) ----------------
        st.markdown("### 👥 Actividad por usuario (productos trabajados)")

        modo_usuarios = st.radio(
            "Periodo para el análisis por usuario",
            ["Solo año seleccionado", "Histórico completo"],
            horizontal=True,
            help="Elige si quieres ver la actividad solo para el año elegido o para toda la historia.",
            key="tab5_modo_usuarios",
        )

        if modo_usuarios == "Solo año seleccionado":
            mask_crea_base = mask_created_year & mask_crea_user_valido
            mask_act_base = mask_updated_year & mask_act_user_valido
            sufijo_titulo = f" en {year}"
        else:
            mask_crea_base = mask_crea_user_valido
            mask_act_base = mask_act_user_valido
            sufijo_titulo = " (histórico completo)"

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

        act_full = pd.merge(
            crea_user,
            act_user,
            on="usuario",
            how="outer",
        ).fillna(0)

        act_full = act_full[
            (act_full["productos_creados"] > 0) | (act_full["productos_actualizados"] > 0)
        ].copy()

        act_full["total_productos"] = (
            act_full["productos_creados"] + act_full["productos_actualizados"]
        )

        if act_full.empty:
            if modo_usuarios == "Solo año seleccionado":
                st.info(f"No hay actividad de usuarios sobre productos en {year}.")
            else:
                st.info("No hay actividad de usuarios sobre productos.")
        else:
            act_top = act_full.sort_values(
                ["total_productos", "productos_creados", "productos_actualizados"],
                ascending=False,
            ).head(20)

            chart_users_crea = (
                alt.Chart(act_top)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "productos_creados:Q",
                        title=f"Productos creados{sufijo_titulo}",
                    ),
                    y=alt.Y("usuario:N", sort="-x", title="Usuario"),
                    tooltip=[
                        alt.Tooltip("usuario:N", title="Usuario"),
                        alt.Tooltip(
                            "productos_creados:Q",
                            title="Creados (ids únicos)",
                        ),
                        alt.Tooltip(
                            "productos_actualizados:Q",
                            title="Actualizados (ids únicos)",
                        ),
                        alt.Tooltip(
                            "total_productos:Q",
                            title="Total productos trabajados",
                        ),
                    ],
                )
                .properties(
                    title=f"Top 20 – Productos creados por usuario{sufijo_titulo}",
                    height=400,
                )
            )

            chart_users_act = (
                alt.Chart(act_top)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "productos_actualizados:Q",
                        title=f"Productos actualizados{sufijo_titulo}",
                    ),
                    y=alt.Y("usuario:N", sort="-x", title="Usuario"),
                    tooltip=[
                        alt.Tooltip("usuario:N", title="Usuario"),
                        alt.Tooltip(
                            "productos_creados:Q",
                            title="Creados (ids únicos)",
                        ),
                        alt.Tooltip(
                            "productos_actualizados:Q",
                            title="Actualizados (ids únicos)",
                        ),
                        alt.Tooltip(
                            "total_productos:Q",
                            title="Total productos trabajados",
                        ),
                    ],
                )
                .properties(
                    title=f"Top 20 – Productos actualizados por usuario{sufijo_titulo}",
                    height=400,
                )
            )

            st.altair_chart(chart_users_crea | chart_users_act, use_container_width=True)

            st.markdown("### 📄 Detalle por usuario")

            tabla_users = (
                act_full.copy()
                .sort_values(
                    ["total_productos", "productos_creados", "productos_actualizados"],
                    ascending=False,
                )
                .rename(
                    columns={
                        "usuario": "Usuario",
                        "productos_creados": "Productos creados",
                        "productos_actualizados": "Productos actualizados",
                        "total_productos": "Total productos trabajados",
                    }
                )[
                    [
                        "Usuario",
                        "Productos creados",
                        "Productos actualizados",
                        "Total productos trabajados",
                    ]
                ]
                .reset_index(drop=True)
            )

            st.dataframe(
                tabla_users.set_index("Usuario"),
                use_container_width=True,
            )
