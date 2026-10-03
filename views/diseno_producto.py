"""Evaluador de Proyectos de Diseño (PDS, AMFE, Ishikawa, QFD).

Portado de `Dise-de-producto/app.py` sin cambios en la logica de calculo ni en
los textos. Los cambios son de estructura:

* Cada modulo metodologico se dibuja en su propia funcion.
* Los resultados se guardan en `st.session_state` para que no desaparezcan al
  interactuar con otras pestanas de la aplicacion unificada.
* Las respuestas de la IA se guardan likewise, para no volver a pagar la
  llamada al proveedor cada vez que se cambia de pestana.
* Las llamadas a la IA muestran en que modelo van: si el elegido falla, la
  funcion `evaluar_texto_llm` reintenta y pasa al siguiente como respaldo.
* Las pestanas son perezosas (`on_change="rerun"`), asi que solo se ejecuta la
  que esta visible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from config_ia import (
    evaluar_texto_llm,
    hay_api_key,
    nombre_modelo_corto,
    resumen_del_intento,
)
from views.comun import boton_calcular, guardar_resultado, leer_resultado

# Claves de resultados de este modulo.
R_PDS = "dise__respuesta_pds"
R_ISHIKAWA = "dise__respuesta_ishikawa"
R_AMFE = "dise__resultado_amfe"
R_QFD = "dise__resultado_qfd"


def _avisar_respaldo(respuesta, ctx: dict) -> None:
    """Avisa si la respuesta vino de otro modelo o tras varios intentos.

    Con el mecanismo de reintento y respaldo, el usuario puede recibir la
    respuesta de un modelo distinto del que eligio. Es importante que lo sepa.
    """
    aviso = resumen_del_intento(respuesta, ctx["model_name"])
    if aviso:
        st.warning(aviso)


# ---------------------------------------------------------------------------
# 1. Evaluador de PDS
# ---------------------------------------------------------------------------
def _evaluar_pds(ctx: dict) -> None:
    st.header("Especificaciones de Diseño de Producto (PDS)")

    with st.expander("ℹ️ ¿Cómo funciona esta evaluación? (Caja Blanca)"):
        st.write(
            """
            **Lo que hace el sistema:**
            El modelo de IA analiza semánticamente tu frase buscando dos cosas críticas:
            1. **Ausencia de ambigüedad:** Penaliza adjetivos subjetivos como "fácil", "resistente", "bonito".
            2. **Presencia de métricas:** Busca activamente números, unidades de medida (kg, mm, W), normativas (ISO, UNE) o condiciones de verificación claras.
            """
        )

    st.write(
        "Introduce una especificación para comprobar si es medible y verificable."
    )
    pds_input = st.text_area(
        "Requisito PDS:",
        placeholder="Ej: La estructura soportará una carga vertical de al menos 2,0 kN...",
        key="dise__pds_entrada",
    )

    if st.button("Evaluar PDS", key="dise__pds_boton", width="stretch"):
        if pds_input:
            prompt_pds = (
                "Eres un profesor de ingeniería. Evalúa si el requisito PDS es"
                " medible, cuantificable y verificable. Si es vago, indícalo. Si es"
                " correcto, felicítalo."
            )
            detalle = st.empty()
            with st.spinner(f"Analizando con {nombre_modelo_corto(ctx)}..."):

                def _progreso(modelo: str, intento: int, total: int) -> None:
                    detalle.caption(f"Probando `{modelo}` (intento {intento}/{total})")

                respuesta = evaluar_texto_llm(
                    ctx, prompt_pds, pds_input, on_progreso=_progreso
                )
            detalle.empty()
            guardar_resultado(R_PDS, (pds_input, respuesta))
        else:
            st.warning("Introduce un requisito.")

    # Se repinta desde session_state, no desde el return del boton, para que el
    # resultado sobreviva al cambio de pestana.
    respuesta = leer_resultado(R_PDS)
    if respuesta is not None:
        texto_analizado, resultado_ia = respuesta
        st.caption(f"Analizando: *{texto_analizado}*")
        st.info(resultado_ia.texto)
        _avisar_respaldo(resultado_ia, ctx)
        if st.button("🔄 Analizar de nuevo", key="dise__pds_repetir"):
            guardar_resultado(R_PDS, None)
            st.rerun()


# ---------------------------------------------------------------------------
# 2. Evaluador de AMFE
# ---------------------------------------------------------------------------
def _evaluar_amfe() -> None:
    st.header("Análisis Modal de Fallos y Efectos (AMFE)")

    with st.expander("ℹ️ ¿Cómo funciona este cálculo? (Caja Blanca)"):
        st.write(
            """
            **Lo que hace el sistema matemáticamente:**
            1. Extrae los valores que has puesto en $S$, $O$ y $D$.
            2. Aplica la fórmula: **$NPR = Severidad \\times Ocurrencia \\times Detección$**.
            3. Evalúa la criticidad mediante lógica condicional:
               - Si la **Severidad es $\\ge 9$** o el **$NPR \\ge 100$**, marca **🔴 Acción Urgente**.
               - Si el **$NPR < 50$**, lo considera **🟢 Riesgo Aceptable**.
               - El resto cae en **🟡 Revisión Normal** o **🟠 Prioridad Media-Alta**.
            """
        )

    st.write(
        "Añade filas según necesites. Pasa el ratón sobre el símbolo **(?)** en"
        " las columnas para ver las instrucciones de puntuación."
    )

    if "df_amfe" not in st.session_state:
        st.session_state.df_amfe = pd.DataFrame(
            {
                "Componente": ["Bisagra", "Tornillo"],
                "Modo de fallo": ["Desgaste", "Aflojamiento"],
                "Severidad (S)": [7, 8],
                "Ocurrencia (O)": [4, 5],
                "Detección (D)": [6, 6],
            }
        )

    config_amfe = {
        "Severidad (S)": st.column_config.NumberColumn(
            "Severidad (S)",
            min_value=1,
            max_value=10,
            help=(
                "Escala 1-10: ¿Cómo de grave sería el fallo? \n1 ="
                " Imperceptible/Sin impacto.\n10 = Catastrófico (riesgo"
                " seguridad/muerte)."
            ),
        ),
        "Ocurrencia (O)": st.column_config.NumberColumn(
            "Ocurrencia (O)",
            min_value=1,
            max_value=10,
            help=(
                "Escala 1-10: ¿Cómo de probable es que ocurra? \n1 = Remoto (muy"
                " poco probable).\n10 = Muy frecuente (ocurre casi siempre)."
            ),
        ),
        "Detección (D)": st.column_config.NumberColumn(
            "Detección (D)",
            min_value=1,
            max_value=10,
            help=(
                "Escala 1-10: ¿Qué tan difícil es de detectar antes del usuario?"
                " \n1 = Casi seguro de detectar (control 100%).\n10 ="
                " Prácticamente imposible."
            ),
        ),
    }

    df_amfe_edit = st.data_editor(
        st.session_state.df_amfe,
        num_rows="dynamic",
        column_config=config_amfe,
        width="stretch",
        key="dise__tabla_amfe",
    )

    if boton_calcular(
        "Calcular Riesgos AMFE", clave="dise__amfe", resultado=R_AMFE
    ):
        df = df_amfe_edit.copy()
        df["NPR Calculado"] = (
            df["Severidad (S)"] * df["Ocurrencia (O)"] * df["Detección (D)"]
        )
        condiciones = [
            (df["Severidad (S)"] >= 9),
            (df["NPR Calculado"] >= 100),
            (df["NPR Calculado"] < 50),
        ]
        valores = [
            "🔴 Acción Urgente (S alta)",
            "🟠 Prioridad Media-Alta",
            "🟢 Riesgo Aceptable",
        ]
        df["Sugerencia Sistema"] = np.select(
            condiciones, valores, default="🟡 Revisión Normal"
        )
        guardar_resultado(R_AMFE, df)

    resultado = leer_resultado(R_AMFE)
    if resultado is not None:
        st.dataframe(resultado, width="stretch")
        st.success("Cálculos verificados correctamente.")


# ---------------------------------------------------------------------------
# 3. Evaluador de Ishikawa
# ---------------------------------------------------------------------------
def _evaluar_ishikawa(ctx: dict) -> None:
    st.header("Diagrama Causa-Efecto (Ishikawa)")

    with st.expander("ℹ️ ¿Cómo funciona esta evaluación? (Caja Blanca)"):
        st.write(
            """
            **Lo que hace el sistema:**
            La IA actúa como un filtro de calidad de redacción técnica. Al leer tu causa, escanea si has usado adjetivos que emiten un "juicio de valor" (ej. ineficiente, desastroso, malo). Si los encuentra, te obligará a reformular la causa hacia un **hecho físico, medible u observable** (ej. "el operario no tiene el manual", "el par de apriete no se verifica").
            """
        )

    st.write(
        "Verifica que las causas identificadas están redactadas como **hechos"
        " verificables** (ej: 'tinta con baja viscosidad') y no como **juicios de"
        " valor** (ej: 'tinta mala')."
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        categoria = st.selectbox(
            "Categoría (6M):",
            [
                "Materiales",
                "Mano de obra",
                "Maquinaria",
                "Métodos",
                "Mantenimiento",
                "Medio ambiente",
            ],
            key="dise__ishikawa_categoria",
        )
    with col2:
        causa_input = st.text_input(
            "Redacción de la causa:",
            placeholder="Ej: Par de apriete no controlado en la línea 2",
            key="dise__ishikawa_entrada",
        )

    if st.button("Evaluar Causa", key="dise__ishikawa_boton", width="stretch"):
        if causa_input:
            prompt_ishikawa = (
                f"Eres un evaluador de calidad. Evalúa esta causa en la categoría"
                f" '{categoria}'. Verifica que no contenga juicios vagos ('malo',"
                " 'desastre'). Debe ser técnico y observable. Si está mal, sugiere"
                " mejora."
            )
            detalle = st.empty()
            with st.spinner("Analizando redacción..."):

                def _progreso(modelo: str, intento: int, total: int) -> None:
                    detalle.caption(f"Probando `{modelo}` (intento {intento}/{total})")

                respuesta_ia = evaluar_texto_llm(
                    ctx, prompt_ishikawa, causa_input, on_progreso=_progreso
                )
            detalle.empty()
            guardar_resultado(
                R_ISHIKAWA, ((categoria, causa_input), respuesta_ia)
            )
        else:
            st.warning("Introduce una causa.")

    respuesta = leer_resultado(R_ISHIKAWA)
    if respuesta is not None:
        (cat, causa), resultado_ia = respuesta
        st.caption(f"Categoría **{cat}** · Causa: *{causa}*")
        st.info(resultado_ia.texto)
        _avisar_respaldo(resultado_ia, ctx)
        if st.button("🔄 Analizar de nuevo", key="dise__ishikawa_repetir"):
            guardar_resultado(R_ISHIKAWA, None)
            st.rerun()


# ---------------------------------------------------------------------------
# 4. Casa de la Calidad (QFD)
# ---------------------------------------------------------------------------
def _evaluar_qfd() -> None:
    st.header("🏠 Casa de la Calidad (QFD)")

    with st.expander("ℹ️ ¿Cómo funciona este cálculo? (Caja Blanca)"):
        st.write(
            """
            **Lo que hace el algoritmo matemáticamente:**
            1. **Suma Producto:** Por cada columna técnica (CÓMO), toma el valor de relación (0, 1, 3 o 9) y lo multiplica por la "Importancia" del QUÉ de esa fila. Luego suma toda la columna para obtener la **Puntuación Absoluta**.
            2. **Proporción:** El valor absoluto por sí solo no tiene un límite máximo (depende de cuántas filas añadas). Lo que importa es la prioridad. Por eso, el sistema suma todas las puntuaciones absolutas de todas las características técnicas para calcular qué **Porcentaje de Importancia Relativa (%)** representa cada una sobre el esfuerzo total del diseño.
            """
        )

    st.write(
        "Construye tu matriz gestionando tus propias columnas (CÓMOs) y filas"
        " (QUÉs)."
    )

    if "df_qfd" not in st.session_state:
        st.session_state.df_qfd = pd.DataFrame(
            {
                "Necesidades (QUÉ)": ["Fácil de limpiar", "Ligero"],
                "Importancia": [5.0, 8.0],
                "CÓMO: Requisito 1": [9, 0],
                "CÓMO: Requisito 2": [0, 9],
            }
        )

    # --- 1. Gestion de columnas (COMOs) ---
    st.markdown("### 🛠️ 1. Gestión de Requisitos Técnicos (Columnas CÓMO)")

    comos_actuales = [
        col for col in st.session_state.df_qfd.columns if col.startswith("CÓMO:")
    ]
    col_add, col_ren, col_del = st.columns(3)

    with col_add:
        st.write("**Añadir nuevo CÓMO**")
        nuevo_como = st.text_input(
            "Nombre de la característica:",
            placeholder="Ej: Rugosidad (µm)",
            key="add_como",
        )
        if st.button("➕ Añadir Columna", key="dise__qfd_anadir"):
            if nuevo_como:
                nombre_columna = f"CÓMO: {nuevo_como}"
                if nombre_columna not in st.session_state.df_qfd.columns:
                    st.session_state.df_qfd[nombre_columna] = 0
                    st.rerun()
                else:
                    st.warning("Esa característica ya existe.")

    with col_ren:
        st.write("**Renombrar un CÓMO**")
        if comos_actuales:
            como_a_renombrar = st.selectbox(
                "Columna a modificar:", comos_actuales, key="sel_renombrar"
            )
            nuevo_nombre = st.text_input(
                "Nuevo nombre:",
                placeholder="Ej: Nivel sonoro (dB)",
                key="ren_como",
            )
            if st.button("✏️ Renombrar Columna", key="dise__qfd_renombrar"):
                if nuevo_nombre:
                    nuevo_nombre_col = f"CÓMO: {nuevo_nombre}"
                    st.session_state.df_qfd.rename(
                        columns={como_a_renombrar: nuevo_nombre_col}, inplace=True
                    )
                    st.rerun()

    with col_del:
        st.write("**Eliminar un CÓMO**")
        if comos_actuales:
            como_a_eliminar = st.selectbox(
                "Columna a borrar:", comos_actuales, key="sel_eliminar"
            )
            if st.button("🗑️ Eliminar Columna", key="dise__qfd_borrar"):
                if len(como_actuales) > 1:
                    st.session_state.df_qfd.drop(
                        columns=[como_a_eliminar], inplace=True
                    )
                    st.rerun()
                else:
                    st.error("⚠️ Debe quedar al menos una columna CÓMO en la matriz.")

    # --- 2. Matriz interactiva ---
    st.markdown("---")
    st.markdown("### 🟦 2. Cuerpo de la Matriz (Relaciones QUÉ - CÓMO)")
    st.write(
        "Puedes **añadir nuevas filas (QUÉs)** haciendo clic en el símbolo `+` en"
        " la parte inferior de la tabla. Pasa el ratón sobre los símbolos **(?)**"
        " para ver la puntuación."
    )

    config_qfd = {
        "Necesidades (QUÉ)": st.column_config.TextColumn(
            "Necesidades (QUÉ)",
            help="Escribe la necesidad en el lenguaje del usuario (Ej: 'Que sea seguro')",
        ),
        "Importancia": st.column_config.NumberColumn(
            "Importancia",
            min_value=0.0,
            max_value=1000.0,
            help=(
                "Valora cuánto le importa esto al cliente. Puedes usar una escala"
                " básica (1-5 o 1-10) o valores avanzados de Importancia"
                " Absoluta. El cálculo de los CÓMOs se ajustará automáticamente"
                " a tu escala."
            ),
        ),
    }

    for col in st.session_state.df_qfd.columns:
        if col.startswith("CÓMO:"):
            config_qfd[col] = st.column_config.NumberColumn(
                col,
                min_value=0,
                max_value=9,
                help=(
                    "Relación entre necesidad y requisito técnico. Valores permitidos:"
                    " \n9 = Fuerte \n3 = Media \n1 = Débil \n0 = Ninguna."
                ),
            )

    df_qfd_edit = st.data_editor(
        st.session_state.df_qfd,
        num_rows="dynamic",
        column_config=config_qfd,
        width="stretch",
        key="dise__tabla_qfd",
    )

    # --- 3. Resultados ---
    st.markdown("### 📊 3. Resultados Técnicos (Base de la Casa)")
    if boton_calcular(
        "Verificar Cálculos QFD y Generar Top-5",
        clave="dise__qfd",
        resultado=R_QFD,
    ):
        try:
            st.session_state.df_qfd = df_qfd_edit

            importancias = df_qfd_edit["Importancia"].fillna(0).astype(float)
            comos_cols = [
                col for col in df_qfd_edit.columns if col.startswith("CÓMO:")
            ]

            resultados_absolutos = {}
            alertas = []

            for col in comos_cols:
                valores_relacion = df_qfd_edit[col].fillna(0).astype(float)
                if not all(valores_relacion.isin([0, 1, 3, 9])):
                    alertas.append(
                        f"⚠️ Atención: En la columna '{col}' se detectaron valores"
                        " distintos a 0, 1, 3 o 9. La metodología exige usar esta"
                        " escala específica."
                    )
                resultados_absolutos[col] = (importancias * valores_relacion).sum()

            df_resultados = pd.DataFrame(
                [resultados_absolutos], index=["Puntuación Absoluta"]
            ).T

            total_absoluto = df_resultados["Puntuación Absoluta"].sum()
            if total_absoluto > 0:
                df_resultados["% Importancia Relativa"] = (
                    df_resultados["Puntuación Absoluta"] / total_absoluto
                ) * 100
                df_resultados["% Importancia Relativa"] = (
                    df_resultados["% Importancia Relativa"].round(1).astype(str) + " %"
                )
            else:
                df_resultados["% Importancia Relativa"] = "0.0 %"

            df_resultados = df_resultados.sort_values(
                by="Puntuación Absoluta", ascending=False
            )

            guardar_resultado(
                R_QFD, {"tabla": df_resultados, "alertas": alertas}
            )
        except Exception as exc:  # noqa: BLE001 - se muestra al usuario
            st.error(
                "Error en el cálculo. Revisa que no hayas introducido texto en"
                f" columnas numéricas. Detalle: {exc}"
            )

    resultado = leer_resultado(R_QFD)
    if resultado is not None:
        for alerta in resultado["alertas"]:
            st.warning(alerta)
        if not resultado["alertas"]:
            st.success(
                "✅ Multiplicaciones verificadas con éxito. Escala 9-3-1-0 respetada."
            )
        st.write(
            "**Ranking Técnico (Priorización de características para el diseño):**"
        )
        st.dataframe(resultado["tabla"], width="stretch")


# ---------------------------------------------------------------------------
# Punto de entrada de la vista
# ---------------------------------------------------------------------------
def render(ctx: dict) -> None:
    """Dibuja el modulo de Diseño de Producto con sus cuatro sub-pestanas."""
    st.markdown(
        "Herramienta de autoevaluación para PDS, AMFE, Ishikawa y QFD."
    )

    if not hay_api_key(ctx):
        st.sidebar.caption("ℹ️ Sin API Key: PDS e Ishikawa no pueden consultar.")

    tab_pds, tab_amfe, tab_ishikawa, tab_qfd = st.tabs(
        ["📋 PDS", "⚠️ AMFE", "🐟 Ishikawa", "🏠 QFD"],
        on_change="rerun",
        key="dise__nav",
    )

    # Solo se ejecuta la sub-pestana visible.
    if tab_pds.open:
        with tab_pds:
            _evaluar_pds(ctx)
    if tab_amfe.open:
        with tab_amfe:
            _evaluar_amfe()
    if tab_ishikawa.open:
        with tab_ishikawa:
            _evaluar_ishikawa(ctx)
    if tab_qfd.open:
        with tab_qfd:
            _evaluar_qfd()