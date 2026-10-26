"""Planificador de Proyectos de Ingenieria (PERT, CPM, Gantt, Zaderenko).

Portado de `planificador-proyectos/app.py` sin cambios en los algoritmos de
calculo. Los cambios son de estructura:

* Cada modo se dibuja en su propia funcion.
* Los resultados se guardan en `st.session_state` para que no desaparezcan al
  interactuar con otras pestanas de la aplicacion unificada.
* Las figuras de matplotlib se cierran tras pintarse.
* Las pestanas son perezosas (`on_change="rerun"`).

Correccion aplicada
-------------------
La linea que mostraba la varianza usaba `f"Varianza ($\\sigma^2$)"`. En una
f-string, `\\s` no es una secuencia de escape valida y Python emitia un
`SyntaxWarning`. Aqui la cadena es normal (no f-string), donde `\\sigma` si es
un escape legitimo.
"""

from __future__ import annotations

from datetime import date

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import scipy.stats as stats
import streamlit as st
from pandas.tseries.offsets import CustomBusinessDay

from views.comun import boton_calcular, guardar_resultado, leer_resultado, mostrar_figura

R_BASICO = "plan__resultado_basico"
R_AVANZADO = "plan__resultado_avanzado"

_DIAS_LABORABLES = CustomBusinessDay(weekmask="Mon Tue Wed Thu Fri")


# ---------------------------------------------------------------------------
# Modo Basico (Activity on Node)
# ---------------------------------------------------------------------------
def _calcular_basico(df: pd.DataFrame, fecha_inicio):
    """Construye el grafo AON y calcula ES/EF/LS/LF, holgura y camino critico.

    Returns
    -------
    dict con los datos ya calculados, listos para pintar.
    """
    grafo = nx.DiGraph()
    for _, row in df.iterrows():
        tarea = str(row["Tarea"]).strip()
        grafo.add_node(tarea, dur=int(row["Duración"]))
        for padre in str(row["Predecesores"]).split(","):
            padre = padre.strip()
            if padre:
                grafo.add_edge(padre, tarea)

    # Si hay un ciclo, topological_sort lanza NetworkXUnfeasible.
    orden = list(nx.topological_sort(grafo))

    es, ef = {}, {}
    for nodo in orden:
        es[nodo] = max([ef[p] for p in grafo.predecessors(nodo)], default=0)
        ef[nodo] = es[nodo] + grafo.nodes[nodo]["dur"]

    duracion_total = max(ef.values())

    ls, lf = {}, {}
    for nodo in reversed(orden):
        sucesores = list(grafo.successors(nodo))
        lf[nodo] = min([ls[s] for s in sucesores], default=duracion_total)
        ls[nodo] = lf[nodo] - grafo.nodes[nodo]["dur"]

    filas = []
    for nodo in orden:
        holgura = lf[nodo] - es[nodo] - grafo.nodes[nodo]["dur"]
        filas.append(
            {
                "Actividad": nodo,
                "d_ij": grafo.nodes[nodo]["dur"],
                "ES": es[nodo],
                "EF": ef[nodo],
                "LS": ls[nodo],
                "LF": lf[nodo],
                "Holgura": holgura,
                "CC": "Sí" if holgura == 0 else "",
            }
        )

    criticos = [n for n in orden if ls[n] - es[n] == 0]
    inicio = pd.to_datetime(fecha_inicio)

    gantt = pd.DataFrame(
        [
            {
                "Tarea": nodo,
                "Inicio": inicio + _DIAS_LABORABLES * int(es[nodo]),
                "Fin": inicio + _DIAS_LABORABLES * int(ef[nodo]),
                "Crítica": "Sí" if nodo in criticos else "No",
            }
            for nodo in orden
        ]
    )

    return {
        "grafo": grafo,
        "orden": orden,
        "criticos": criticos,
        "duracion_total": duracion_total,
        "tabla": pd.DataFrame(filas),
        "gantt": gantt,
    }


def _pintar_basico(res: dict, fecha_inicio) -> None:
    inicio = pd.to_datetime(fecha_inicio)
    fecha_fin = inicio + _DIAS_LABORABLES * int(res["duracion_total"])

    st.success(
        f"**Duración total:** {res['duracion_total']} días laborables | "
        f"**Fecha de finalización:** {fecha_fin.strftime('%d/%m/%Y')}"
    )
    st.dataframe(res["tabla"], width="stretch")

    col_gantt, col_grafo = st.columns(2)

    with col_gantt:
        fig_gantt = px.timeline(
            res["gantt"],
            x_start="Inicio",
            x_end="Fin",
            y="Tarea",
            color="Crítica",
            color_discrete_map={"Sí": "#e74c3c", "No": "#3498db"},
        )
        fig_gantt.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_gantt, width="stretch")

    with col_grafo:
        fig, ax = plt.subplots(figsize=(6, 4))
        posiciones = nx.spring_layout(res["grafo"], seed=42)
        colores = [
            "#e74c3c" if n in res["criticos"] else "#3498db"
            for n in res["grafo"].nodes()
        ]
        nx.draw(
            res["grafo"],
            posiciones,
            with_labels=True,
            node_size=2000,
            node_color=colores,
            font_weight="bold",
            edge_color="gray",
            arrows=True,
            ax=ax,
        )
        mostrar_figura(fig)


def _modo_basico() -> None:
    st.markdown("### 1. Datos de Entrada (Modo Básico)")

    datos_basicos = {
        "Tarea": ["A", "B", "C", "D", "E", "F"],
        "Duración": [2, 3, 3, 4, 3, 5],
        "Predecesores": ["", "A", "A", "B", "C", "D,E"],
    }
    df_basico = pd.DataFrame(datos_basicos)
    df_editado = st.data_editor(
        df_basico, num_rows="dynamic", width="stretch", key="plan__tabla_basica"
    )

    fecha_inicio = st.date_input(
        "Selecciona la fecha de inicio del proyecto:",
        date.today(),
        key="plan__fecha_basica",
    )

    if boton_calcular(
        "Calcular Proyecto Básico", clave="plan__basico", resultado=R_BASICO
    ):
        try:
            guardar_resultado(
                R_BASICO, _calcular_basico(df_editado, fecha_inicio)
            )
        except nx.NetworkXUnfeasible:
            st.error("Error: Bucle detectado en las precedencias.")

    resultado = leer_resultado(R_BASICO)
    if resultado is not None:
        _pintar_basico(resultado, fecha_inicio)


# ---------------------------------------------------------------------------
# Modo Avanzado (Activity on Arrow / Zaderenko)
# ---------------------------------------------------------------------------
def _calcular_avanzado(df: pd.DataFrame, fecha_inicio):
    """Calcula t_e, varianzas, tiempos胡萝卜, camino critico y probabilidad."""
    grafo = nx.DiGraph()
    for _, row in df.iterrows():
        actividad = str(row["Actividad"]).strip()
        i, j = int(row["i (Origen)"]), int(row["j (Destino)"])
        t_opt = float(row["t_opt"])
        t_prob = float(row["t_prob"])
        t_pes = float(row["t_pes"])

        t_e = (t_opt + 4 * t_prob + t_pes) / 6
        varianza = ((t_pes - t_opt) / 6) ** 2
        grafo.add_edge(i, j, act=actividad, te=t_e, var=varianza)

    nodos = list(nx.topological_sort(grafo))

    t_early = {n: 0 for n in nodos}
    for n in nodos:
        for p in grafo.predecessors(n):
            t_early[n] = max(t_early[n], t_early[p] + grafo[p][n]["te"])

    duracion_proy = t_early[nodos[-1]]

    t_last = {n: duracion_proy for n in nodos}
    for n in reversed(nodos):
        for s in grafo.successors(n):
            t_last[n] = min(t_last[n], t_last[s] - grafo[n][s]["te"])

    filas = []
    var_proy = 0.0
    for u, v, datos in grafo.edges(data=True):
        t_e = datos["te"]
        es, ef = t_early[u], t_early[u] + t_e
        lf, ls = t_last[v], t_last[v] - t_e
        holgura = lf - ef
        es_critica = abs(holgura) < 1e-5

        grafo[u][v]["is_critical"] = es_critica
        if es_critica:
            var_proy += datos["var"]

        filas.append(
            {
                "Actividad": datos["act"],
                "i-j": f"{u}-{v}",
                "te": round(t_e, 2),
                "ti": round(es, 2),
                "tj": round(ef, 2),
                "ti*": round(ls, 2),
                "tj*": round(t_last[v], 2),
                "Hsi": round(t_last[u] - t_early[u], 2),
                "Hsj": round(t_last[v] - t_early[v], 2),
                "HTij": round(holgura, 2),
                "CC": "CC" if es_critica else "",
            }
        )

    # Matriz de Zaderenko.
    # El original mezclaba floats con "" en las celdas y ademas usaba etiquetas
    # de nodo numericas junto a las de texto ("t_i*", "t_j"). Con eso pyarrow no
    # conseguia serializar la tabla ("Could not convert 't_i*' with type str:
    # tried to convert to int64") y Streamlit tenia que parchearla sola.
    # Aqui todas las celdas y todas las etiquetas son texto, de modo que la
    # tabla es homogenea y se serializa sin avisos. El aspecto no cambia.
    etiquetas = [str(n) for n in nodos]
    matriz = pd.DataFrame("", index=etiquetas, columns=etiquetas, dtype=object)
    for u, v, datos in grafo.edges(data=True):
        matriz.loc[str(u), str(v)] = f"{datos['te']:.2f}"
    matriz.loc["t_i*"] = [f"{t_last[n]:.2f}" for n in nodos]
    matriz["t_j"] = [f"{t_early[n]:.2f}" for n in nodos] + [""]

    gantt = pd.DataFrame(
        [
            {
                "Actividad": f["Actividad"],
                "Inicio": pd.to_datetime(fecha_inicio)
                + _DIAS_LABORABLES * int(f["ti"]),
                "Fin": pd.to_datetime(fecha_inicio)
                + _DIAS_LABORABLES * int(f["tj"]),
                "Crítica": "Sí" if f["CC"] == "CC" else "No",
            }
            for f in filas
            if f["te"] > 0
        ]
    )

    return {
        "grafo": grafo,
        "nodos": nodos,
        "duracion": duracion_proy,
        "varianza": var_proy,
        "tabla": pd.DataFrame(filas),
        "matriz": matriz,
        "gantt": gantt,
    }


def _pintar_avanzado(res: dict, fecha_inicio) -> None:
    inicio = pd.to_datetime(fecha_inicio)
    fecha_fin = inicio + _DIAS_LABORABLES * int(res["duracion"])

    st.success(
        f"**Duración total esperada ($T_e$):** {res['duracion']:.2f} días "
        f"laborables | **Fecha de finalización:** {fecha_fin.strftime('%d/%m/%Y')}"
    )

    col_izq, col_der = st.columns(2)

    with col_izq:
        st.markdown("**Matriz de Zaderenko**")
        st.dataframe(res["matriz"], width="stretch")

        st.markdown("**Tabla de Resultados**")
        st.dataframe(res["tabla"], width="stretch")

        st.markdown("**Probabilidad de Cumplimiento (Estadística)**")
        desviacion = np.sqrt(res["varianza"])
        # Cadena normal (no f-string) para que \sigma sea un escape valido.
        st.write(
            "Varianza ($\\sigma^2$): "
            f"{res['varianza']:.2f} | "
            f"Desviación ($\\sigma$): {desviacion:.2f}"
        )
        plazo_objetivo = st.number_input(
            "Plazo objetivo (días laborables):",
            value=float(res["duracion"]) + 1,
            key="plan__plazo_objetivo",
        )
        if desviacion > 0:
            probabilidad = (
                stats.norm.cdf((plazo_objetivo - res["duracion"]) / desviacion) * 100
            )
            st.info(f"Probabilidad de éxito: **{probabilidad:.2f}%**")

    with col_der:
        st.markdown("**Diagrama PERT (Sucesos y Actividades)**")
        fig, ax = plt.subplots(figsize=(8, 5))
        grafo = res["grafo"]
        posiciones = nx.spring_layout(grafo, seed=42)

        nx.draw_networkx_nodes(
            grafo,
            posiciones,
            node_size=1200,
            node_color="#f1f2f6",
            edgecolors="black",
            ax=ax,
        )
        nx.draw_networkx_labels(
            grafo, posiciones, font_size=10, font_weight="bold", ax=ax
        )
        colores_arista = [
            "red" if grafo[u][v]["is_critical"] else "gray"
            for u, v in grafo.edges()
        ]
        nx.draw_networkx_edges(
            grafo, posiciones, edge_color=colores_arista, arrows=True, ax=ax
        )
        etiquetas = {(u, v): grafo[u][v]["act"] for u, v in grafo.edges()}
        nx.draw_networkx_edge_labels(
            grafo, posiciones, edge_labels=etiquetas, font_color="blue", ax=ax
        )
        mostrar_figura(fig)

        st.markdown("**Diagrama de Gantt**")
        fig_gantt = px.timeline(
            res["gantt"],
            x_start="Inicio",
            x_end="Fin",
            y="Actividad",
            color="Crítica",
            color_discrete_map={"Sí": "#e74c3c", "No": "#3498db"},
        )
        fig_gantt.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_gantt, width="stretch")


def _modo_avanzado() -> None:
    st.markdown("### 1. Datos de Actividades Sucesos (i $\\rightarrow$ j)")

    datos_avanzados = {
        "Actividad": ["A", "B", "C", "D", "B'", "C'"],
        "i (Origen)": [1, 1, 2, 2, 3, 4],
        "j (Destino)": [2, 3, 4, 5, 5, 5],
        "t_opt": [1, 2, 3, 6, 0, 0],
        "t_prob": [1, 2, 3, 6, 0, 0],
        "t_pes": [1, 2, 3, 6, 0, 0],
    }
    df_avanz = pd.DataFrame(datos_avanzados)
    df_editado = st.data_editor(
        df_avanz, num_rows="dynamic", width="stretch", key="plan__tabla_avanzada"
    )

    fecha_inicio = st.date_input(
        "Selecciona la fecha de inicio del proyecto:",
        date.today(),
        key="plan__fecha_avanzada",
    )

    if boton_calcular(
        "Calcular Proyecto Avanzado",
        clave="plan__avanzado",
        resultado=R_AVANZADO,
    ):
        try:
            guardar_resultado(
                R_AVANZADO, _calcular_avanzado(df_editado, fecha_inicio)
            )
        except nx.NetworkXUnfeasible:
            st.error("Error estructural en el grafo.")

    resultado = leer_resultado(R_AVANZADO)
    if resultado is not None:
        _pintar_avanzado(resultado, fecha_inicio)


# ---------------------------------------------------------------------------
# Modo EVM (Earned Value Management / Gestión del Valor Ganado)
# ---------------------------------------------------------------------------

def _modo_evm() -> None:
    """Ventana de EVM con cálculos en vivo, explicaciones (?) y tabla editable."""
    st.markdown("### 📊 Gestión del Valor Ganado (EVM)")

    with st.expander("❓ Conceptos Básicos de EVM (¿Qué es cada variable?)", expanded=True):
        st.markdown(
            """
            - **BAC** (Budget at Completion): Presupuesto total aprobado para el proyecto.
            - **PV** (Planned Value): Valor que debería haberse ganado según lo planificado.
            - **EV** (Earned Value): Valor realmente ganado con el trabajo realizado.
            - **AC** (Actual Cost): Coste real incurrido hasta la fecha.
            - **SV** = EV − PV: Variación del cronograma (positivo = adelantado).
            - **CV** = EV − AC: Variación del coste (positivo = por debajo de presupuesto).
            - **SPI** = EV / PV: Índice de desempeño del cronograma (>1 adelantado, <1 retrasado).
            - **CPI** = EV / AC: Índice de desempeño del coste (>1 por debajo, <1 por encima).
            - **EAC** = AC + (BAC − EV) / CPI (fórmula típica): Estimación del coste total.
            - **ETC** = EAC − AC: Coste restante estimado para terminar.
            - **VAC** = BAC − EAC: Variación estimada al finalizar el proyecto.
            - **TCPI** = (BAC − EV) / (BAC − AC): Índice de desempeño necesario para cumplir BAC.
            """
        )

    st.markdown("---")
    st.markdown("### ⚙️ Parámetros Generales")
    col_bac, col_per = st.columns(2)
    with col_bac:
        bac = st.number_input(
            "BAC (Budget at Completion) €",
            min_value=0.0,
            value=100000.0,
            step=5000.0,
            format="%.0f",
            key="evm__bac",
            help="Presupuesto total aprobado. Se actualiza todo al cambiar este número.",
        )
    with col_per:
        num_periodos = st.number_input(
            "Número de períodos (filas en tabla)",
            min_value=1,
            max_value=30,
            value=5,
            step=1,
            key="evm__num_per",
        )

    st.markdown("### 📋 Datos por Período (Edita cada fila)")
    st.caption(
        "Introduce % Planificado y % Real Completado. El sistema calcula PV, EV, AC y todos los índices."
    )

    # Generador rápido: rellena con valores lineales o constantes.
    col_gen_plan, col_gen_real, col_gen_ac = st.columns(3)
    with col_gen_plan:
        plan_lineal = st.checkbox(
            "% Planificado lineal (20% por período)", value=True, key="evm__plan_lin"
        )
    with col_gen_real:
        real_lineal = st.checkbox(
            "% Real lineal (20% por período)", value=False, key="evm__real_lin"
        )
    with col_gen_ac:
        ac_lineal = st.checkbox(
            "AC lineal (20% de BAC por período)", value=False, key="evm__ac_lin"
        )

    # Datos iniciales
    datos_iniciales = {
        "Período": list(range(1, int(num_periodos) + 1)),
        "% Planificado Acum.": [
            round(100 / int(num_periodos) * i, 2) for i in range(1, int(num_periodos) + 1)
        ] if plan_lineal else [0.0] * int(num_periodos),
        "% Real Completado Acum.": [
            round(100 / int(num_periodos) * i, 2) for i in range(1, int(num_periodos) + 1)
        ] if real_lineal else [0.0] * int(num_periodos),
        "AC Acumulado (€)": [
            round(bac / int(num_periodos) * i, 2) for i in range(1, int(num_periodos) + 1)
        ] if ac_lineal else [0.0] * int(num_periodos),
    }
    df_inicial = pd.DataFrame(datos_iniciales)

    df_editado = st.data_editor(
        df_inicial,
        column_config={
            "Período": st.column_config.NumberColumn("Período", disabled=True),
            "% Planificado Acum.": st.column_config.NumberColumn(
                "% Planificado Acum.", min_value=0.0, max_value=100.0, step=1.0, format="%.2f"
            ),
            "% Real Completado Acum.": st.column_config.NumberColumn(
                "% Real Completado Acum.", min_value=0.0, max_value=100.0, step=1.0, format="%.2f"
            ),
            "AC Acumulado (€)": st.column_config.NumberColumn(
                "AC Acumulado (€)", min_value=0.0, format="%.2f"
            ),
        },
        num_rows="fixed",
        hide_index=True,
        width="stretch",
        key="evm__tabla",
    )

    # Cálculos en vivo (no dependen de botón)
    resultados_filas = []
    pv_total = ev_total = ac_total = 0.0
    # Tomamos los valores del último período como referencia acumulada total
    if not df_editado.empty:
        pv_total = float(bac) * float(df_editado["% Planificado Acum."].iloc[-1]) / 100.0
        ev_total = float(bac) * float(df_editado["% Real Completado Acum."].iloc[-1]) / 100.0
        ac_total = float(df_editado["AC Acumulado (€)"].iloc[-1])

    for _, row in df_editado.iterrows():
        periodo = int(row["Período"])
        plan_pct = float(row["% Planificado Acum."])
        real_pct = float(row["% Real Completado Acum."])
        ac = float(row["AC Acumulado (€)"])

        pv = float(bac) * plan_pct / 100.0
        ev = float(bac) * real_pct / 100.0

        sv = ev - pv
        cv = ev - ac
        spi = ev / pv if pv > 0 else 0.0
        cpi = ev / ac if ac > 0 else 0.0

        resultados_filas.append({
            "Período": periodo,
            "% Plan": plan_pct,
            "% Real": real_pct,
            "PV (€)": pv,
            "EV (€)": ev,
            "AC (€)": ac,
            "SV (€)": sv,
            "CV (€)": cv,
            "SPI": spi,
            "CPI": cpi,
        })

    df_resultados = pd.DataFrame(resultados_filas)
    df_resultados = df_resultados.round({
        "PV (€)": 2, "EV (€)": 2, "AC (€)": 2,
        "SV (€)": 2, "CV (€)": 2, "SPI": 3, "CPI": 3,
    })

    # Variables finales para métricas globales (último período)
    if not df_resultados.empty:
        pv_final = float(df_resultados["PV (€)"].iloc[-1])
        ev_final = float(df_resultados["EV (€)"].iloc[-1])
        ac_final = float(df_resultados["AC (€)"].iloc[-1])
        sv_final = float(df_resultados["SV (€)"].iloc[-1])
        cv_final = float(df_resultados["CV (€)"].iloc[-1])
        spi_final = float(df_resultados["SPI"].iloc[-1]) if float(df_resultados["PV (€)"].iloc[-1]) > 0 else 0.0
        cpi_final = float(df_resultados["CPI"].iloc[-1]) if float(df_resultados["AC (€)"].iloc[-1]) > 0 else 0.0
    else:
        pv_final = ev_final = ac_final = sv_final = cv_final = spi_final = cpi_final = 0.0

    # Proyecciones (último período como referencia acumulada)
    bac_val = float(bac)
    ev_final_val = ev_final
    ac_final_val = ac_final
    cpi_final_val = cpi_final

    eac = ac_final_val + (bac_val - ev_final_val) / cpi_final_val if cpi_final_val > 0 else 0.0
    etc = eac - ac_final_val
    vac = bac_val - eac
    tcpi = (bac_val - ev_final_val) / (bac_val - ac_final_val) if (bac_val - ac_final_val) != 0 else 0.0

    # ------------------------------------------------------------------
    # Apartado A: Variables Básicas (con signos ?)
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("1. Variables Básicas de EVM")

    col_a1, col_a2, col_a3, col_a4 = st.columns(4)
    with col_a1:
        with st.expander("❓ ¿Cómo se calcula BAC?"):
            st.markdown("**BAC** = Presupuesto total aprobado. Es un dato fijo del proyecto.")
        st.metric("BAC (€)", f"{bac_val:,.2f}")
    with col_a2:
        with st.expander("❓ ¿Cómo se calcula PV?"):
            st.markdown("**PV** = BAC × (% Planificado / 100). Es lo que debería haberse ganado según el plan.")
        st.metric("PV Final (€)", f"{pv_final:,.2f}")
    with col_a3:
        with st.expander("❓ ¿Cómo se calcula EV?"):
            st.markdown("**EV** = BAC × (% Real Completado / 100). Es lo que realmente se ha ganado con el trabajo hecho.")
        st.metric("EV Final (€)", f"{ev_final:,.2f}")
    with col_a4:
        with st.expander("❓ ¿Cómo se calcula AC?"):
            st.markdown("**AC** = Coste real acumulado que se ha gastado hasta este período. Se introduce directamente en la tabla.")
        st.metric("AC Final (€)", f"{ac_final:,.2f}")

    # ------------------------------------------------------------------
    # Apartado B: Índices de Rendimiento (con signos ?)
    # ------------------------------------------------------------------
    st.subheader("2. Índices de Rendimiento (SPI / CPI)")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        with st.expander("❓ ¿Cómo se calcula SPI?"):
            st.markdown(
                "**SPI** = EV / PV. Si SPI > 1, el proyecto va adelantado respecto al cronograma. "
                "Si SPI < 1, va retrasado. Si SPI = 1, está exactamente en el plan."
            )
        st.metric("SPI (Último período)", f"{spi_final:.3f}")
        if spi_final > 1:
            st.info("Adelantado respecto al cronograma.")
        elif spi_final < 1:
            st.warning("Retrasado respecto al cronograma.")
        else:
            st.success("En plan respecto al cronograma.")

    with col_b2:
        with st.expander("❓ ¿Cómo se calcula CPI?"):
            st.markdown(
                "**CPI** = EV / AC. Si CPI > 1, se está gastando menos de lo previsto (buena eficiencia). "
                "Si CPI < 1, se gasta más de lo previsto. Si CPI = 1, costó exactamente lo que valió."
            )
        st.metric("CPI (Último período)", f"{cpi_final:.3f}")
        if cpi_final > 1:
            st.info("Por debajo del presupuesto (buena eficiencia).")
        elif cpi_final < 1:
            st.warning("Por encima del presupuesto (baja eficiencia).")
        else:
            st.success("Coste igual al valor ganado.")

    # ------------------------------------------------------------------
    # Apartado C: Variaciones (SV / CV)
    # ------------------------------------------------------------------
    st.subheader("3. Variaciones (SV / CV)")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        with st.expander("❓ ¿Cómo se calcula SV (Schedule Variance)?"):
            st.markdown("**SV** = EV − PV. SV > 0 indica que se ha ganado más valor del previsto (adelantado); SV < 0 indica retraso.")
        st.metric("SV Final (€)", f"{sv_final:,.2f}")
        if sv_final > 0:
            st.info("Adelantado (EV > PV).")
        elif sv_final < 0:
            st.warning("Retrasado (EV < PV).")
        else:
            st.success("Exactamente en plan.")

    with col_c2:
        with st.expander("❓ ¿Cómo se calcula CV (Cost Variance)?"):
            st.markdown("**CV** = EV − AC. CV > 0 indica que el valor ganado supera lo gastado (ahorro); CV < 0 indica sobrecoste.")
        st.metric("CV Final (€)", f"{cv_final:,.2f}")
        if cv_final > 0:
            st.info("Ahorro respecto al presupuesto.")
        elif cv_final < 0:
            st.warning("Sobrecoste respecto al presupuesto.")
        else:
            st.success("Coste igual al valor ganado.")

    # ------------------------------------------------------------------
    # Apartado D: Proyecciones (EAC / ETC / VAC / TCPI)
    # ------------------------------------------------------------------
    st.subheader("4. Proyecciones y Estimaciones (EAC / ETC / VAC / TCPI)")
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    with col_d1:
        with st.expander("❓ ¿Cómo se calcula EAC?"):
            st.markdown(
                "**EAC** = AC + (BAC − EV) / CPI. Es la estimación del coste total "
                "si el rendimiento futuro (CPI) sigue igual que hasta ahora."
            )
        st.metric("EAC (€)", f"{eac:,.2f}")
    with col_d2:
        with st.expander("❓ ¿Cómo se calcula ETC?"):
            st.markdown("**ETC** = EAC − AC. Es lo que falta por gastar (estimado) para completar el proyecto.")
        st.metric("ETC (€)", f"{etc:,.2f}")
    with col_d3:
        with st.expander("❓ ¿Cómo se calcula VAC?"):
            st.markdown("**VAC** = BAC − EAC. Es la diferencia estimada al finalizar: positivo = ahorro, negativo = sobrecoste estimado.")
        st.metric("VAC (€)", f"{vac:,.2f}")
        if vac > 0:
            st.info("Se espera terminar por debajo del presupuesto.")
        elif vac < 0:
            st.warning("Se espera terminar por encima del presupuesto.")
        else:
            st.success("Se espera terminar exactamente en presupuesto.")
    with col_d4:
        with st.expander("❓ ¿Cómo se calcula TCPI?"):
            st.markdown(
                "**TCPI** = (BAC − EV) / (BAC − AC). Índice de desempeño que debe mantenerse "
                "en el trabajo restante para cumplir exactamente con el BAC."
            )
        st.metric("TCPI", f"{tcpi:.3f}")
        if tcpi > cpi_final:
            st.info("Necesita mejorar el rendimiento para cumplir BAC.")
        elif tcpi < cpi_final:
            st.info("Con este rendimiento, terminará por debajo de BAC.")
        else:
            st.info("Rendimiento actual exactamente suficiente para cumplir BAC.")

    # ------------------------------------------------------------------
    # Apartado E: Tabla Completa de Desglose (como Viabilidad)
    # ------------------------------------------------------------------
    st.subheader("5. Desglose Completo por Período")
    st.markdown(
        "Esta tabla muestra todos los datos editados y los cálculos derivados por período. "
        "Se actualiza automáticamente al modificar cualquier fila."
    )

    # Añadimos datos acumulados al dataframe para la tabla final
    df_desglose = df_resultados.copy()
    # Añadir datos editados originales para referencia
    df_desglose["% Real (Editado)"] = df_editado["% Real Completado Acum."].values
    df_desglose["AC (Editado) €"] = df_editado["AC Acumulado (€)"].values

    # Reordenar columnas para que sea legible
    orden_columnas = [
        "Período", "% Plan", "% Real", "PV (€)", "EV (€)", "AC (€)",
        "SV (€)", "CV (€)", "SPI", "CPI",
        "% Real (Editado)", "AC (Editado) €",
    ]
    df_desglose_final = df_desglose[[c for c in orden_columnas if c in df_desglose.columns]]
    st.dataframe(
        df_desglose_final.style.format({
            "PV (€)": "{:.2f} €",
            "EV (€)": "{:.2f} €",
            "AC (€)": "{:.2f} €",
            "SV (€)": "{:.2f} €",
            "CV (€)": "{:.2f} €",
            "SPI": "{:.3f}",
            "CPI": "{:.3f}",
            "% Plan": "{:.2f} %",
            "% Real": "{:.2f} %",
            "% Real (Editado)": "{:.2f} %",
            "AC (Editado) €": "{:.2f} €",
        }).map(lambda x: "background-color: #e8f5e9" if isinstance(x, (int, float)) and x > 0 else ("background-color: #ffebee" if isinstance(x, (int, float)) and x < 0 else ""), subset=["SV (€)", "CV (€)"]),
        width="stretch",
        hide_index=True,
        use_container_width=True,
    )

    # ------------------------------------------------------------------
    # Apartado F: Gráfico de Evolución (PV / EV / AC)
    # ------------------------------------------------------------------
    st.subheader("6. Evolución Gráfica de PV, EV y AC")
    st.markdown("Comparación visual del valor planificado, ganado y gastado a lo largo de los períodos.")

    fig, ax = plt.subplots(figsize=(10, 5))
    periodos = df_resultados["Período"].tolist()
    pv_vals = df_resultados["PV (€)"].tolist()
    ev_vals = df_resultados["EV (€)"].tolist()
    ac_vals = df_resultados["AC (€)"].tolist()

    ax.plot(periodos, pv_vals, marker="o", label="PV (Planificado)", color="#3498db", linewidth=2)
    ax.plot(periodos, ev_vals, marker="s", label="EV (Ganado)", color="#2ecc71", linewidth=2)
    ax.plot(periodos, ac_vals, marker="^", label="AC (Coste Real)", color="#e74c3c", linewidth=2)
    ax.set_xlabel("Período")
    ax.set_ylabel("Valor (€)")
    ax.set_title("Evolución de PV, EV y AC por Período")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(periodos)

    # Añadir etiquetas de puntos con los valores
    for i, p in enumerate(periodos):
        ax.annotate(f"{pv_vals[i]:.0f}", (p, pv_vals[i]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7, color="#3498db")
        ax.annotate(f"{ev_vals[i]:.0f}", (p, ev_vals[i]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7, color="#2ecc71")
        ax.annotate(f"{ac_vals[i]:.0f}", (p, ac_vals[i]), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=7, color="#e74c3c")

    mostrar_figura(fig)

    # ------------------------------------------------------------------
    # Footer / Resumen final interactivo
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📌 Resumen Interactivo del Proyecto")
    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("BAC", f"{bac_val:,.0f} €")
    col_r2.metric("EAC Estimado", f"{eac:,.0f} €", delta=f"{vac:,.0f} €" if vac != 0 else None)
    col_r3.metric("SPI / CPI Final", f"{spi_final:.2f} / {cpi_final:.2f}")

    if bac_val > 0 and ac_final_val > 0:
        if cpi_final > 1 and spi_final > 1:
            st.success("✅ Proyecto adelantado y con ahorro estimado respecto al presupuesto.")
        elif cpi_final > 1 and spi_final < 1:
            st.info("ℹ️ Proyecto con ahorro estimado, pero retrasado respecto al cronograma.")
        elif cpi_final < 1 and spi_final > 1:
            st.warning("⚠️ Proyecto adelantado, pero con sobrecoste estimado.")
        elif cpi_final < 1 and spi_final < 1:
            st.error("❌ Proyecto retrasado y con sobrecoste estimado. Requiere acción correctiva.")
        else:
            st.info("Proyecto en equilibrio respecto a plan, pero con rendimiento cercano a 1.")


# ---------------------------------------------------------------------------
# Punto de entrada de la vista (modificado para incluir EVM)
# ---------------------------------------------------------------------------
def render(_ctx: dict) -> None:
    """Dibuja el Planificador con sus dos modos."""
    st.markdown(
        "Planificación, análisis y seguimiento de proyectos de ingeniería "
        "mediante PERT, CPM, Gantt y método de Zaderenko."
    )

    tab_basico, tab_avanzado, tab_evm = st.tabs(
        [
            "🟢 Modo Básico (Actividades y Precedencias)",
            "🔵 Modo Avanzado (Nodos y Zaderenko)",
            "📈 EVM (Gestión del Valor Ganado)",
        ],
        on_change="rerun",
        key="plan__nav",
    )

    if tab_basico.open:
        with tab_basico:
            _modo_basico()
    if tab_avanzado.open:
        with tab_avanzado:
            _modo_avanzado()
    if tab_evm.open:
        with tab_evm:
            _modo_evm()