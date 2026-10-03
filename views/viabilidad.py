"""Calculadora de Viabilidad de Proyectos (VAN, TIR, Payback).

Portado de `calculadora-viabilidad-proyectos/calculadora_app.py` sin cambios
en los calculos financieros. Los cambios son de estructura:

* Los parametros generales se dibujan dentro de la propia pestana en lugar de
  la barra lateral. Es el unico cambio de disposicion de toda la aplicacion
  unificada, y es necesario: la barra lateral queda reservada a la
  configuracion de IA, que es comun a toda la app y se rellena una sola vez.
* `use_container_width` (obsoleto, con fecha de eliminacion superada) se
  sustituye por `width="stretch"`.

La app original recalculaba en cada pulsacion, sin boton, asi que aqui tambien
se recalcula en vivo: no necesita persistir resultados en session_state.
"""

from __future__ import annotations

import numpy as np
import numpy_financial as npf
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render(_ctx: dict) -> None:
    """Dibuja la calculadora de viabilidad."""
    st.markdown(
        "Ajusta los parámetros generales y **modifica los flujos de caja año a año**."
    )

    with st.expander("ℹ️ Ayuda y Conceptos Básicos (Tasa de Descuento vs. TIR)"):
        st.markdown(
            """
    **La Tasa de Descuento (El "Listón"):**
    Es la rentabilidad mínima exigida. Depende de factores externos (inflación, riesgo, coste del préstamo).

    **La TIR - Tasa Interna de Retorno (El "Salto"):**
    Es la rentabilidad real interna que genera el diseño.

    *Regla de oro:* Si la **TIR** es mayor que la **Tasa de Descuento**, el proyecto es financieramente viable.
    """
        )

    st.markdown("---")
    st.markdown("### ⚙️ Parámetros Generales")

    col_inv, col_tasa, col_vida = st.columns(3)
    with col_inv:
        inversion = st.number_input(
            "Inversión Inicial (€)",
            min_value=1000,
            max_value=1000000,
            value=50000,
            step=500,
            key="viab__inversion",
        )
    with col_tasa:
        tasa_descuento = st.number_input(
            "Tasa de Descuento (%)",
            min_value=0.1,
            max_value=25.0,
            value=5.0,
            step=0.1,
            key="viab__tasa",
        )
    with col_vida:
        vida_util = st.number_input(
            "Vida Útil (Años)",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            key="viab__vida",
        )

    st.markdown("### ⚡ Generador Rápido")
    flujo_base = st.number_input(
        "Flujo Base Anual (€)",
        min_value=-50000,
        max_value=500000,
        value=12000,
        step=100,
        help=(
            "Cambia este valor para rellenar toda la tabla. Luego edita años "
            "concretos a mano."
        ),
        key="viab__flujo_base",
    )

    # 1. Tabla de flujos de caja
    st.subheader("1. Edición de Flujos de Caja")
    df_inicial = pd.DataFrame(
        {
            "Año": range(1, int(vida_util) + 1),
            "Flujo de Caja (€)": [flujo_base] * int(vida_util),
        }
    )

    df_editado = st.data_editor(
        df_inicial,
        column_config={
            "Año": st.column_config.NumberColumn("Año", disabled=True),
            "Flujo de Caja (€)": st.column_config.NumberColumn(
                "Flujo de Caja (€)",
                min_value=-1000000,
                max_value=10000000,
                step=100,
            ),
        },
        hide_index=True,
        width="stretch",
        key="viab__tabla_flujos",
    )

    # 2. Calculos
    flujos_variables = df_editado["Flujo de Caja (€)"].tolist()
    flujos_totales = [-inversion] + flujos_variables
    tasa = tasa_descuento / 100
    anios = list(range(int(vida_util) + 1))

    flujos_actualizados = [
        f / ((1 + tasa) ** t) for t, f in zip(anios, flujos_totales)
    ]
    flujos_acumulados = np.cumsum(flujos_totales).tolist()
    flujos_acumulados_actualizados = np.cumsum(flujos_actualizados).tolist()

    # Payback con interpolacion lineal.
    payback = None
    for i, acumulado in enumerate(flujos_acumulados):
        if acumulado >= 0:
            if i == 0:
                payback = 0.0
            else:
                acumulado_anterior = abs(flujos_acumulados[i - 1])
                flujo_del_anio = flujos_totales[i]
                fraccion_anio = (
                    acumulado_anterior / flujo_del_anio if flujo_del_anio != 0 else 0
                )
                payback = (i - 1) + fraccion_anio
            break

    van = sum(flujos_actualizados)
    # El original llamaba a npf.irr() dos veces en la misma expresion; se
    # guarda el resultado para no repetir el calculo.
    tir_bruto = npf.irr(flujos_totales)
    tir = tir_bruto * 100 if tir_bruto is not None else 0

    # 3. Resultados
    st.markdown("---")
    st.subheader("2. Resultados Económicos")
    col1, col2, col3 = st.columns(3)
    col1.metric(label="VAN", value=f"{van:,.2f} €")
    col2.metric(label="TIR", value=f"{tir:.2f} %")
    col3.metric(
        label="Payback",
        value=f"{payback:.2f} Años" if payback is not None else "No rentable",
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=anios,
            y=flujos_acumulados,
            mode="lines+markers",
            name="Flujo Acumulado",
            line=dict(color="#1f77b4", width=3),
        )
    )
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="red",
        annotation_text="Punto de Equilibrio",
    )
    fig.update_layout(
        title="Evolución de los Flujos de Caja Acumulados",
        xaxis_title="Años del Proyecto",
        yaxis_title="Flujo Acumulado (€)",
        template="plotly_white",
    )
    st.plotly_chart(fig, width="stretch")

    # 4. Desglose
    st.subheader("3. Desglose Financiero Completo")
    st.markdown(
        "Esta tabla incluye el **Año 0 (Inversión Inicial)** y muestra cómo se "
        "descuenta el dinero a valor presente."
    )

    df_resultados = pd.DataFrame(
        {
            "Año": anios,
            "Flujo Neto (€)": flujos_totales,
            "Flujo Acumulado (€)": flujos_acumulados,
            "Flujo Actualizado (€)": flujos_actualizados,
            "Flujo Acumulado Actualizado (€)": flujos_acumulados_actualizados,
        }
    )

    st.dataframe(
        df_resultados.style.format(
            {
                "Flujo Neto (€)": "{:,.2f} €",
                "Flujo Acumulado (€)": "{:,.2f} €",
                "Flujo Actualizado (€)": "{:,.2f} €",
                "Flujo Acumulado Actualizado (€)": "{:,.2f} €",
            }
        ),
        width="stretch",
        hide_index=True,
    )