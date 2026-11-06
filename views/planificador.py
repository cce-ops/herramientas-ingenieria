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

_DIAS_ES = {
    "Mon": "lun",
    "Tue": "mar",
    "Wed": "mié",
    "Thu": "jue",
    "Fri": "vie",
    "Sat": "sáb",
    "Sun": "dom",
}


def _fmt_num(valor) -> str:
    """Formatea un numero sin decimales innecesarios (1, 2.5, 1.556...)."""
    valor = float(valor)
    if abs(valor - round(valor)) < 1e-9:
        return str(int(round(valor)))
    return f"{valor:.3f}".rstrip("0").rstrip(".")




# ---------------------------------------------------------------------------
# Modo Basico (Activity on Node)
# ---------------------------------------------------------------------------
def _calcular_basico(df: pd.DataFrame, fecha_inicio, usar_pert: bool = False):
    """Construye el grafo AON y calcula los tiempos PERT/AOA.

    Devuelve, ademas del grafo AON, el grafo AOA (actividades como aristas) y
    la tabla de holgura con el formato de los apuntes:
    Actividad (i-j), Nombre, dij, ti, tj, t*i, t*j, Hi, Hj, HTij, CC.
    """
    grafo = nx.DiGraph()
    for _, row in df.iterrows():
        tarea = str(row["Tarea"]).strip()
        if not tarea:
            continue
        if usar_pert:
            t_opt = float(row["t_opt"])
            t_prob = float(row["t_prob"])
            t_pes = float(row["t_pes"])
            te = (t_opt + 4 * t_prob + t_pes) / 6
            var = ((t_pes - t_opt) / 6) ** 2
            grafo.add_node(
                tarea, dur=te, var=var, t_opt=t_opt, t_prob=t_prob, t_pes=t_pes
            )
        else:
            dur = float(row["Duración"])
            grafo.add_node(
                tarea, dur=dur, var=0.0, t_opt=dur, t_prob=dur, t_pes=dur
            )
        for padre in str(row["Predecesores"]).split(","):
            padre = padre.strip()
            if padre:
                grafo.add_edge(padre, tarea)

    orden = list(nx.topological_sort(grafo))
    duraciones = {n: grafo.nodes[n]["dur"] for n in orden}

    # --- Tiempos AON (para el calendario y el Gantt) ---
    es, ef = {}, {}
    for nodo in orden:
        es[nodo] = max([ef[p] for p in grafo.predecessors(nodo)], default=0.0)
        ef[nodo] = es[nodo] + grafo.nodes[nodo]["dur"]

    duracion_total = max(ef.values()) if ef else 0.0

    ls, lf = {}, {}
    for nodo in reversed(orden):
        sucesores = list(grafo.successors(nodo))
        lf[nodo] = min([ls[s] for s in sucesores], default=duracion_total)
        ls[nodo] = lf[nodo] - grafo.nodes[nodo]["dur"]

    criticos = [
        n for n in orden
        if abs(lf[n] - es[n] - grafo.nodes[n]["dur"]) < 1e-5
    ]
    var_proy = sum(grafo.nodes[n].get("var", 0.0) for n in criticos)

    for u, v in grafo.edges():
        grafo[u][v]["is_critical"] = (u in criticos and v in criticos)

    # --- Conversion AON -> AOA y tabla de holgura de los apuntes ---
    grafo_aoa, mapping = _aon_to_aoa(grafo.copy(), duraciones)
    nodos_aoa = list(nx.topological_sort(grafo_aoa))

    t_early = {n: 0.0 for n in nodos_aoa}
    for n in nodos_aoa:
        for p in grafo_aoa.predecessors(n):
            t_early[n] = max(
                t_early[n], t_early[p] + grafo_aoa[p][n].get("te", 0.0)
            )

    t_last = {}
    for n in nodos_aoa:
        if grafo_aoa.out_degree(n) == 0:
            t_last[n] = t_early[n]
        else:
            t_last[n] = float("inf")
    for n in reversed(nodos_aoa):
        sucesores = list(grafo_aoa.successors(n))
        if sucesores:
            t_last[n] = min(
                t_last[s] - grafo_aoa[n][s].get("te", 0.0) for s in sucesores
            )

    filas = []
    for u, v, datos in grafo_aoa.edges(data=True):
        dij = datos.get("te", 0.0)
        ti = t_early[u]
        tj = t_early[v]
        ti_star = t_last[u]
        tj_star = t_last[v]
        filas.append(
            {
                "Actividad": f"{u}-{v}",
                "Nombre": datos.get("act", ""),
                "dij": dij,
                "ti": ti,
                "tj": tj,
                "t*i": ti_star,
                "t*j": tj_star,
                "Hi": ti_star - ti,
                "Hj": tj_star - tj,
                "HTij": tj_star - ti - dij,
                "CC": "CC" if abs(tj_star - ti - dij) < 1e-5 else "-",
                "_orden": (ti, datos.get("act", "")),
            }
        )
    filas.sort(key=lambda f: f["_orden"])
    for fila in filas:
        fila.pop("_orden", None)

    inicio = pd.to_datetime(fecha_inicio)
    gantt = pd.DataFrame(
        [
            {
                "Tarea": nodo,
                "Inicio": inicio + _DIAS_LABORABLES * int(round(es[nodo])),
                "Fin": inicio + _DIAS_LABORABLES * int(round(ef[nodo])),
                "Crítica": "Sí" if nodo in criticos else "No",
            }
            for nodo in orden
        ]
    )

    return {
        "grafo": grafo,
        "aoa": grafo_aoa,
        "nodos_aoa": nodos_aoa,
        "early": t_early,
        "last": t_last,
        "orden": orden,
        "es": es,
        "ef": ef,
        "ls": ls,
        "lf": lf,
        "criticos": criticos,
        "duracion_total": duracion_total,
        "varianza": var_proy,
        "tabla": pd.DataFrame(filas),
        "gantt": gantt,
        "usar_pert": usar_pert,
    }


def _aon_to_aoa(grafo_aon, duraciones):
    """Convierte un grafo AON (actividades como nodos) a AOA (PERT: actividades como aristas).
    Devuelve (grafo_aoa, mapping) donde grafo_aoa es un DiGraph con nodos=sucesos y aristas=actividades.
    Los numeros de sucesos reflejan el orden temporal (EARLY time).
    """
    # Topological sort
    orden = list(nx.topological_sort(grafo_aon))

    # Calcular EARLY times para cada actividad (usando duracion)
    early = {}
    for act in orden:
        preds = list(grafo_aon.predecessors(act))
        if not preds:
            early[act] = 0.0
        else:
            early[act] = max(early[p] + duraciones.get(p, 0) for p in preds)

    # Ordenar actividades por EARLY time para asignar numeros de sucesos
    # (mayor EARLY primero para que el dominante obtenga numero mayor)
    orden_por_early = sorted(orden, key=lambda a: (early[a], -duraciones.get(a, 0)))

    suceso_inicio = {}
    suceso_fin = {}
    siguiente_suceso = 1

    # Primera pasada: asignar sucesos de inicio Y fin (en orden de EARLY, menor duración primero)
    for act in sorted(orden, key=lambda a: (early[a], duraciones.get(a, 0))):
        preds = list(grafo_aon.predecessors(act))
        if not preds:
            suceso_inicio[act] = siguiente_suceso
            siguiente_suceso += 1
            suceso_fin[act] = siguiente_suceso
            siguiente_suceso += 1
        elif len(preds) == 1:
            suceso_inicio[act] = suceso_fin[preds[0]]
            suceso_fin[act] = siguiente_suceso
            siguiente_suceso += 1
        else:
            # Multiples predecesores: el dominante (mayor duracion) se conecta directamente.
            dominante = max(preds, key=lambda p: duraciones.get(p, 0))
            suceso_inicio[act] = suceso_fin[dominante]
            # Crear ficticias para los no-dominantes
            for p in preds:
                if p != dominante:
                    # Crear suceso intermedio
                    suceso_intermedio = siguiente_suceso
                    siguiente_suceso += 1
                    # La ficticia ira de suceso_fin[p] a suceso_intermedio, y de suceso_intermedio a suceso_inicio[act]
                    # Pero para simplificar, añadimos una sola ficticia de suceso_fin[p] a suceso_inicio[act]
                    pass  # se añadira en la construccion del grafo AOA

    # Segunda pasada: reasignar sucesos de fin para que reflejen la duracion.
    # Procesar en orden de (EARLY, -duración) para que el dominante obtenga numero mayor.
    for act in sorted(orden, key=lambda a: (early[a], -duraciones.get(a, 0))):
        if act not in suceso_fin:
            suceso_fin[act] = siguiente_suceso
            siguiente_suceso += 1
        else:
            # Si ya tiene suceso_fin, verificar que sea mayor que el suceso de inicio
            if suceso_fin[act] <= suceso_inicio[act]:
                suceso_fin[act] = suceso_inicio[act] + 1
                siguiente_suceso = max(siguiente_suceso, suceso_fin[act] + 1)

    # Construir grafo AOA
    grafo_aoa = nx.DiGraph()
    nodos_aoa = set()
    for act in orden:
        i = suceso_inicio[act]
        j = suceso_fin[act]
        nodos_aoa.add(i)
        nodos_aoa.add(j)
        grafo_aoa.add_edge(i, j, act=act, te=duraciones[act], dur=duraciones[act])
        preds = list(grafo_aon.predecessors(act))
        if len(preds) > 1:
            dominante = max(preds, key=lambda p: duraciones.get(p, 0))
            for p in preds:
                if p != dominante:
                    fi = suceso_fin[p]
                    fj = suceso_inicio[act]
                    if fi != fj:
                        grafo_aoa.add_edge(fi, fj, act=f"Fict_{p}", te=0, dur=0, ficticia=True)

    # Renumerar nodos para que sean consecutivos (1, 2, 3, ...)
    mapping = {n: i+1 for i, n in enumerate(sorted(nodos_aoa))}
    grafo_aoa = nx.relabel_nodes(grafo_aoa, mapping)

    return grafo_aoa, mapping




def _pintar_basico(res: dict, fecha_inicio) -> None:
    if "aoa" not in res:
        st.warning(
            "El resultado guardado es de una version anterior. "
            "Vuelve a pulsar **Calcular Proyecto Básico**."
        )
        return

    inicio = pd.to_datetime(fecha_inicio)
    dur_total = res["duracion_total"]
    fecha_fin = inicio + _DIAS_LABORABLES * int(round(dur_total))

    st.success(
        f"**Duración total:** {_fmt_num(dur_total)} días laborables | "
        f"**Fecha de finalización:** {fecha_fin.strftime('%d/%m/%Y')}"
    )

    grafo_aoa = res["aoa"]
    nodos_aoa = res["nodos_aoa"]
    t_early = res["early"]
    t_last = res["last"]
    usar_pert = res.get("usar_pert", False)

    # --- 1. Diagrama PERT (sucesos y actividades) ---
    st.markdown("**1. Diagrama PERT (sucesos y actividades)**")
    fig_pert, ax_pert = plt.subplots(figsize=(10, 6))

    niveles_topo = {}
    for n in grafo_aoa.nodes():
        pred = list(grafo_aoa.predecessors(n))
        nivel = 0 if not pred else max(niveles_topo.get(p, 0) for p in pred) + 1
        niveles_topo[n] = nivel

    por_nivel = {}
    for n, nv in niveles_topo.items():
        por_nivel.setdefault(nv, []).append(n)

    pos_aoa = {}
    for nv, nodos in por_nivel.items():
        # El suceso con mayor tiempo EARLY queda arriba (como en los apuntes)
        for idx, n in enumerate(sorted(nodos, key=lambda x: -t_early[x])):
            pos_aoa[n] = (nv * 3.0, -idx * 2.0)

    def _es_critica_aoa(u, v):
        dij = grafo_aoa[u][v].get("te", 0.0)
        return (
            abs(t_early[u] + dij - t_early[v]) < 1e-5
            and abs(t_last[v] - t_early[v]) < 1e-5
        )

    colores_nodo = [
        "#e74c3c" if any(_es_critica_aoa(u, v) for u, v in grafo_aoa.edges() if u == n or v == n) else "#3498db"
        for n in grafo_aoa.nodes()
    ]
    nx.draw_networkx_nodes(
        grafo_aoa, pos_aoa, node_size=1600, node_color=colores_nodo,
        edgecolors="black", linewidths=2, ax=ax_pert,
    )
    etiquetas_nodo = {
        n: f"{n}\nE={_fmt_num(t_early[n])}\nL={_fmt_num(t_last[n])}"
        for n in grafo_aoa.nodes()
    }
    nx.draw_networkx_labels(
        grafo_aoa, pos_aoa, labels=etiquetas_nodo, font_size=7,
        font_weight="bold", ax=ax_pert,
    )
    for u, v in grafo_aoa.edges():
        critica = _es_critica_aoa(u, v)
        ficticia = grafo_aoa[u][v].get("dur", 1) == 0
        nx.draw_networkx_edges(
            grafo_aoa, pos_aoa, edgelist=[(u, v)],
            edge_color="red" if critica else "gray",
            width=3 if critica else 1.5,
            arrows=True, arrowstyle="->", arrowsize=15, ax=ax_pert,
            style="dashed" if ficticia else "solid",
            connectionstyle="arc3,rad=0.05",
        )
    etiquetas = {}
    for u, v in grafo_aoa.edges():
        act = grafo_aoa[u][v].get("act", "")
        dur = grafo_aoa[u][v].get("te", 0)
        etiquetas[(u, v)] = f"{act}\n{_fmt_num(dur)}"
    nx.draw_networkx_edge_labels(
        grafo_aoa, pos_aoa, edge_labels=etiquetas, font_color="blue",
        ax=ax_pert, font_size=8,
    )
    ax_pert.axis("off")
    mostrar_figura(fig_pert)

    # --- 2. Matriz de Zaderenko ---
    st.markdown("**2. Matriz de Zaderenko**")
    etiquetas_filas = [f"{_fmt_num(t_early[n])}   {n}" for n in nodos_aoa]
    columnas = [str(n) for n in nodos_aoa]
    matriz_z = pd.DataFrame("", index=etiquetas_filas, columns=columnas)
    matriz_z.index.name = "ti · i / j"
    for u, v, datos in grafo_aoa.edges(data=True):
        matriz_z.loc[f"{_fmt_num(t_early[u])}   {u}", str(v)] = _fmt_num(
            datos.get("te", 0)
        )
    matriz_z.loc["t*i/j"] = [_fmt_num(t_last[n]) for n in nodos_aoa]
    st.dataframe(matriz_z, width="stretch")
    st.caption(
        "Filas: tiempo EARLY (ti) y suceso i · Columnas: suceso j · "
        "Última fila: tiempos LAST (t*i/j)."
    )

    # --- 3. Cálculo de la holgura y camino crítico ---
    st.markdown("**3. Cálculo de la Holgura y Determinación del Camino Crítico**")
    tabla = res["tabla"]
    st.dataframe(
        tabla.style.format(
            {
                "dij": "{:g}",
                "ti": "{:g}",
                "tj": "{:g}",
                "t*i": "{:g}",
                "t*j": "{:g}",
                "Hi": "{:g}",
                "Hj": "{:g}",
                "HTij": "{:g}",
            }
        ),
        width="stretch",
        hide_index=True,
        column_config={
            "Actividad": st.column_config.TextColumn(
                "Actividad", help="Suceso de origen i y suceso de destino j (i-j)."
            ),
            "Nombre": st.column_config.TextColumn(
                "Nombre", help="Nombre de la actividad."
            ),
            "dij": st.column_config.NumberColumn(
                "dij", help="Duración de la actividad (días)."
            ),
            "ti": st.column_config.NumberColumn(
                "ti", help="Tiempo EARLY del suceso de origen i."
            ),
            "tj": st.column_config.NumberColumn(
                "tj", help="Tiempo EARLY del suceso de destino j."
            ),
            "t*i": st.column_config.NumberColumn(
                "t*i", help="Tiempo LAST del suceso de origen i."
            ),
            "t*j": st.column_config.NumberColumn(
                "t*j", help="Tiempo LAST del suceso de destino j."
            ),
            "Hi": st.column_config.NumberColumn(
                "Hi", help="Holgura del suceso de origen. Fórmula: Hi = t*i − ti."
            ),
            "Hj": st.column_config.NumberColumn(
                "Hj", help="Holgura del suceso de destino. Fórmula: Hj = t*j − tj."
            ),
            "HTij": st.column_config.NumberColumn(
                "HTij",
                help=(
                    "Holgura total de la actividad. "
                    "Fórmula: HTij = t*j − ti − dij. Si HTij = 0, es crítica."
                ),
            ),
            "CC": st.column_config.TextColumn(
                "CC", help="Camino Crítico: CC si HTij = 0; en caso contrario, -."
            ),
        },
    )
    st.caption(
        "Holgura del suceso: Hi = t*i − ti ; Hj = t*j − tj. "
        "Holgura total de la actividad: HTij = t*j − ti − dij."
    )

    # --- 4. Tiempos esperados y varianzas PERT ---
    st.markdown("**4. Tiempos Esperados y Varianzas PERT**")
    st.caption("te = (to + 4·tm + tp) / 6 ; σ² = [(tp − to) / 6]²")
    if not usar_pert:
        st.caption(
            "Modo Duración fija: se toma to = tm = tp = dij. "
            "Cambia a «PERT (t_opt / t_prob / t_pes)» para introducir los "
            "tiempos optimista, más probable y pesimista del enunciado."
        )
    grafo = res["grafo"]
    filas_pert = []
    for n in res["orden"]:
        nodo = grafo.nodes[n]
        filas_pert.append(
            {
                "Actividad": n,
                "tm": nodo.get("t_prob", nodo.get("dur", 0.0)),
                "to": nodo.get("t_opt", nodo.get("dur", 0.0)),
                "tp": nodo.get("t_pes", nodo.get("dur", 0.0)),
                "te": nodo.get("dur", 0.0),
                "σ²": nodo.get("var", 0.0),
            }
        )
    df_pert = pd.DataFrame(filas_pert)
    st.dataframe(
        df_pert.style.format(
            {
                "tm": "{:g}",
                "to": "{:g}",
                "tp": "{:g}",
                "te": "{:.2f}",
                "σ²": "{:.3f}",
            }
        ),
        width="stretch",
        hide_index=True,
        column_config={
            "Actividad": st.column_config.TextColumn(
                "Actividad", help="Nombre de la actividad."
            ),
            "tm": st.column_config.NumberColumn(
                "tm", help="Tiempo más probable (caso normal)."
            ),
            "to": st.column_config.NumberColumn(
                "to", help="Tiempo optimista (mejor caso)."
            ),
            "tp": st.column_config.NumberColumn(
                "tp", help="Tiempo pesimista (peor caso)."
            ),
            "te": st.column_config.NumberColumn(
                "te", help="Tiempo esperado. Fórmula: te = (to + 4·tm + tp) / 6."
            ),
            "σ²": st.column_config.NumberColumn(
                "σ²", help="Varianza. Fórmula: σ² = [(tp − to) / 6]²."
            ),
        },
    )
    camino = " – ".join(res["criticos"])
    sigma_proy = np.sqrt(res["varianza"]) if res["varianza"] > 0 else 0.0
    st.markdown(
        f"**Camino crítico:** {camino} &nbsp;&nbsp; "
        f"**Duración esperada = {_fmt_num(dur_total)} días**"
    )
    st.markdown(
        f"**Varianza del camino crítico = {res['varianza']:.3f}** ; "
        f"σ = {sigma_proy:.3f} días"
    )

    # --- 5. Calendario de Ejecución ---
    st.markdown("**5. Calendario de Ejecución**")

    def _celda_calendario(valor):
        fecha = inicio + _DIAS_LABORABLES * int(round(valor))
        return (
            f"{_fmt_num(valor)}\n"
            f"{_DIAS_ES[fecha.strftime('%a')]} {fecha.strftime('%d/%m')}"
        )

    filas_cal = []
    for n in res["orden"]:
        filas_cal.append(
            {
                "Actividad": n,
                "ES\nInicio temprano": _celda_calendario(res["es"][n]),
                "EF\nFin temprano": _celda_calendario(res["ef"][n]),
                "LS\nInicio tardío": _celda_calendario(res["ls"][n]),
                "LF\nFin tardío": _celda_calendario(res["lf"][n]),
            }
        )
    st.dataframe(
        pd.DataFrame(filas_cal),
        width="stretch",
        hide_index=True,
        column_config={
            "Actividad": st.column_config.TextColumn(
                "Actividad", help="Nombre de la actividad."
            ),
            "ES\nInicio temprano": st.column_config.TextColumn(
                "ES · Inicio temprano",
                help="Early Start: valor PERT y fecha calendario.",
            ),
            "EF\nFin temprano": st.column_config.TextColumn(
                "EF · Fin temprano",
                help="Early Finish: valor PERT y fecha calendario.",
            ),
            "LS\nInicio tardío": st.column_config.TextColumn(
                "LS · Inicio tardío",
                help="Late Start: valor PERT y fecha calendario.",
            ),
            "LF\nFin tardío": st.column_config.TextColumn(
                "LF · Fin tardío",
                help="Late Finish: valor PERT y fecha calendario.",
            ),
        },
    )
    st.caption(
        "Supuesto: se trabaja de lunes a viernes. Las fechas de fin son el "
        "momento en que la actividad queda terminada. "
        f"Fecha prevista de finalización del proyecto: "
        f"{_DIAS_ES[fecha_fin.strftime('%a')]} {fecha_fin.strftime('%d/%m/%Y')}."
    )

    # --- 6. Diagrama de Gantt ---
    st.markdown("**6. Diagrama de Gantt**")
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

    # --- 7. Probabilidad de Cumplimiento ---
    st.markdown("**7. Probabilidad de Cumplimiento**")
    st.markdown(
        "La duración del proyecto se aproxima mediante una distribución normal "
        "asociada al camino crítico esperado."
    )
    if not usar_pert:
        st.info(
            "En modo **Duración fija** no hay incertidumbre: σ² = 0. "
            "Cambia a **PERT (t_opt / t_prob / t_pes)** para obtener la tabla "
            "de probabilidad del enunciado."
        )
    var = res["varianza"]
    mu = dur_total
    sigma = np.sqrt(var) if var > 0 else 0.0
    st.markdown(
        f"**Camino crítico:** {camino} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"**μ = {_fmt_num(mu)} días** &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"**σ² = {var:.3f}** &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"**σ = {sigma:.3f} días**"
    )
    st.markdown("Para una fecha objetivo T*: t = (T* − μ) / σ ; P(cumplir) = Φ(t)")

    # Tabla resumen: to, tm, tp y varianza por actividad critica y del proyecto
    filas_resumen = []
    for etiqueta, clave in (
        ("Optimista to", "t_opt"),
        ("Más probable tm", "t_prob"),
        ("Pesimista tp", "t_pes"),
    ):
        fila = {"Duración (días)": etiqueta}
        for n in res["criticos"]:
            nodo = grafo.nodes[n]
            fila[n] = _fmt_num(nodo.get(clave, nodo.get("dur", 0.0)))
        fila["Proyecto"] = _fmt_num(
            sum(
                grafo.nodes[n].get(clave, grafo.nodes[n].get("dur", 0.0))
                for n in res["criticos"]
            )
        )
        filas_resumen.append(fila)

    fila_var = {"Duración (días)": "Varianza σ²"}
    for n in res["criticos"]:
        fila_var[n] = f"{grafo.nodes[n].get('var', 0.0):.2f}"
    fila_var["Proyecto"] = f"{var:.2f}"
    filas_resumen.append(fila_var)

    df_resumen = pd.DataFrame(filas_resumen).set_index("Duración (días)")
    st.dataframe(df_resumen, width="stretch")

    # Tabla de plazos objetivo y probabilidad
    filas_plazo = []
    base = int(round(mu))
    for plazo in range(base - 1, base + 5):
        fecha = inicio + _DIAS_LABORABLES * plazo
        if sigma > 0:
            t_valor = (plazo - mu) / sigma
            probabilidad = stats.norm.cdf(t_valor)
        else:
            t_valor = 0.0
            probabilidad = 1.0 if plazo >= mu else 0.0
        filas_plazo.append(
            {
                "Plazo T*": plazo,
                "Fecha calendario": (
                    f"{_DIAS_ES[fecha.strftime('%a')]} {fecha.strftime('%d/%m')}"
                ),
                "t = (T* − μ) / σ": f"{t_valor:.2f}",
                "P(t < T*)": f"{probabilidad * 100:.2f} %",
            }
        )
    st.dataframe(pd.DataFrame(filas_plazo), width="stretch", hide_index=True)

    prob_dos_dias = (
        stats.norm.cdf(2 / sigma) * 100 if sigma > 0 else 100.0
    )
    st.markdown(
        "**Lectura:** terminar en la fecha esperada tiene una probabilidad del "
        "50 %. Con dos días laborables adicionales, la probabilidad sube "
        f"aproximadamente al {prob_dos_dias:.0f} %."
    )

    with st.expander("Consulta rápida de otro plazo"):
        plazo_objetivo = st.number_input(
            "Plazo objetivo (días laborables):",
            value=float(base) + 1,
            key="plan__plazo_objetivo_basico",
        )
        if sigma > 0:
            probabilidad = stats.norm.cdf((plazo_objetivo - mu) / sigma) * 100
        else:
            probabilidad = 100.0 if plazo_objetivo >= mu else 0.0
        st.info(f"Probabilidad de éxito: **{probabilidad:.2f}%**")


def _modo_basico() -> None:
    st.markdown("### 1. Datos de Entrada (Modo Básico)")

    modo_pert = st.selectbox(
        "Tipo de duración:",
        ["Duración fija (días)", "PERT (t_opt / t_prob / t_pes)"],
        index=0,
        key="plan__modo_pert_basico",
        help="PERT usa tiempos optimista (to), más probable (tm) y pesimista (tp) para calcular t_e = (to + 4tm + tp)/6.",
    )
    usar_pert = modo_pert.startswith("PERT")

    if usar_pert:
        datos_basicos = {
            "Tarea": ["A", "B", "C", "D", "E"],
            "t_opt": [1, 1, 3, 2, 4],
            "t_prob": [1, 2, 5, 3, 7],
            "t_pes": [1, 3, 7, 4, 10],
            "Predecesores": ["", "A", "B", "B", "C,D"],
        }
    else:
        datos_basicos = {
            "Tarea": ["A", "B", "C", "D", "E"],
            "Duración": [1, 2, 5, 3, 7],
            "Predecesores": ["", "A", "B", "B", "C,D"],
        }
    df_basico = pd.DataFrame(datos_basicos)

    if usar_pert:
        df_editado = st.data_editor(
            df_basico,
            num_rows="dynamic",
            width="stretch",
            key="plan__tabla_basica_pert",
            column_config={
                "Tarea": st.column_config.TextColumn("Tarea"),
                "t_opt": st.column_config.NumberColumn("t_opt (optimista)", min_value=0.0, step=0.5),
                "t_prob": st.column_config.NumberColumn("t_prob (probable)", min_value=0.0, step=0.5),
                "t_pes": st.column_config.NumberColumn("t_pes (pesimista)", min_value=0.0, step=0.5),
                "Predecesores": st.column_config.TextColumn("Predecesores"),
            },
        )
    else:
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
                R_BASICO, _calcular_basico(df_editado, fecha_inicio, usar_pert)
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
        if "Duración" in df.columns:
            dur = float(row["Duración"])
            t_opt = t_prob = t_pes = dur
        else:
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

    # Nodos criticos: nodos que aparecen en alguna arista critica
    criticos = set()
    for u, v, datos in grafo.edges(data=True):
        if datos.get("is_critical", False):
            criticos.add(u)
            criticos.add(v)
    return {
        "grafo": grafo,
        "nodos": nodos,
        "duracion": duracion_proy,
        "varianza": var_proy,
        "tabla": pd.DataFrame(filas),
        "matriz": matriz,
        "gantt": gantt,
        "criticos": list(criticos),
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
        fig, ax = plt.subplots(figsize=(10, 6))
        grafo = res["grafo"]

        # Layout PERT clasico: nodos ordenados por nivel topologico (izquierda a derecha)
        # Nivel 0 = nodos sin predecesores, nivel N = nodos sin sucesores
        niveles_topo = {}
        for n in grafo.nodes():
            pred = list(grafo.predecessors(n))
            nivel = 0 if not pred else max(niveles_topo.get(p, 0) for p in pred) + 1
            niveles_topo[n] = nivel
        # Agrupar por nivel para separar verticalmente
        por_nivel = {}
        for n, nv in niveles_topo.items():
            por_nivel.setdefault(nv, []).append(n)
        posiciones = {}
        for nv, nodos in por_nivel.items():
            for idx, n in enumerate(nodos):
                posiciones[n] = (nv * 3.0, -idx * 2.0)

        # Colores de nodos: rojo si critico
        colores_nodo = [
            "#e74c3c" if n in res["criticos"] else "#3498db"
            for n in grafo.nodes()
        ]
        nx.draw_networkx_nodes(
            grafo, posiciones, node_size=1800, node_color=colores_nodo,
            edgecolors="black", linewidths=2, ax=ax,
        )

        # Etiquetas de nodos: numero + E (EARLY) y L (LAST)
        t_early_map = {}
        t_last_map = {}
        if "tabla" in res:
            for _, row in res["tabla"].iterrows():
                parts = str(row["i-j"]).split("-")
                if len(parts) == 2:
                    u, v = parts[0], parts[1]
                    t_early_map[u] = row["ti"]
                    t_early_map[v] = row["tj"]
                    t_last_map[u] = row["ti*"]
                    t_last_map[v] = row["tj*"]
        for n in grafo.nodes():
            if n not in t_early_map:
                t_early_map[n] = 0.0
                t_last_map[n] = 0.0

        etiquetas_nodo = {}
        for n in grafo.nodes():
            te = t_early_map.get(n, 0)
            tl = t_last_map.get(n, 0)
            etiquetas_nodo[n] = f"{n}\nE={te:.0f}\nL={tl:.0f}"
        nx.draw_networkx_labels(
            grafo, posiciones, labels=etiquetas_nodo, font_size=7, font_weight="bold", ax=ax,
        )

        # Colores y grosor de aristas
        colores_arista = [
            "red" if (u in res["criticos"] and v in res["criticos"]) else "gray"
            for u, v in grafo.edges()
        ]
        anchos = [3 if (u in res["criticos"] and v in res["criticos"]) else 1.5 for u, v in grafo.edges()]
        # Ficticias (dur=0) como discontinuas
        estilos = ["dashed" if grafo[u][v].get("te", 1) == 0 else "solid" for u, v in grafo.edges()]
        for i, (u, v) in enumerate(grafo.edges()):
            nx.draw_networkx_edges(
                grafo, posiciones, edgelist=[(u, v)], edge_color=colores_arista[i],
                width=anchos[i], arrows=True, arrowstyle="->", arrowsize=15, ax=ax,
                style=estilos[i], connectionstyle="arc3,rad=0.05",
            )

        # Etiquetas de aristas: nombre actividad + duracion
        etiquetas = {}
        for u, v in grafo.edges():
            act = grafo[u][v].get("act", "")
            dur = grafo[u][v].get("te", "")
            if dur == "":
                dur = grafo.nodes[u].get("dur", "")
            etiquetas[(u, v)] = f"{act}\n{dur:.1f}" if isinstance(dur, float) else f"{act}\n{dur}"
        nx.draw_networkx_edge_labels(
            grafo, posiciones, edge_labels=etiquetas, font_color="blue", ax=ax,
            font_size=8,
        )
        ax.axis("off")
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

    modo_dur = st.selectbox(
        "Tipo de duración:",
        ["Duración fija (días)", "PERT (t_opt / t_prob / t_pes)"],
        index=0,
        key="plan__modo_dur_avanzado",
        help="Si eliges 'Duración fija', se usa la misma duración para t_opt, t_prob y t_pes.",
    )
    usar_pert_av = modo_dur.startswith("PERT")

    if usar_pert_av:
        datos_avanzados = {
            "Actividad": ["A", "B", "C", "D", "B'", "C'"],
            "i (Origen)": [1, 1, 2, 2, 3, 4],
            "j (Destino)": [2, 3, 4, 5, 5, 5],
            "t_opt": [1, 2, 3, 6, 0, 0],
            "t_prob": [1, 2, 3, 6, 0, 0],
            "t_pes": [1, 2, 3, 6, 0, 0],
        }
    else:
        datos_avanzados = {
            "Actividad": ["A", "B", "C", "D", "B'", "C'"],
            "i (Origen)": [1, 1, 2, 2, 3, 4],
            "j (Destino)": [2, 3, 4, 5, 5, 5],
            "Duración": [1, 2, 3, 6, 0, 0],
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

def _modo_evm_estandar() -> None:
    """Ventana de EVM con cálculos en vivo, explicaciones (?) y tabla editable."""
    st.markdown("### 📊 Gestión del Valor Ganado (EVM)")

    with st.expander("❓ Conceptos Básicos de EVM (¿Qué es cada variable?)", expanded=True):
        st.markdown(
            """
            - **BAC** (Budget at Completion): Presupuesto total aprobado para el proyecto.
            - **PV** (Planned Value): Valor que debería haberse ganado según lo planificado.
            - **EV** (Earned Value): Valor realmente ganado con el trabajo realizado.
            - **AC** (Actual Cost): Coste real incurrido hasta la fecha.
            - **SV** = EV - PV: Variación del cronograma (positivo = adelantado).
            - **CV** = EV - AC: Variación del coste (positivo = por debajo de presupuesto).
            - **SPI** = EV / PV: Índice de desempeño del cronograma (>1 adelantado, <1 retrasado).
            - **CPI** = EV / AC: Índice de desempeño del coste (>1 por debajo, <1 por encima).
            - **EAC** = AC + (BAC - EV) / CPI (fórmula típica): Estimación del coste total.
            - **ETC** = EAC - AC: Coste restante estimado para terminar.
            - **VAC** = BAC - EAC: Variación estimada al finalizar el proyecto.
            - **TCPI** = (BAC - EV) / (BAC - AC): Índice de desempeño necesario para cumplir BAC.
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
            st.markdown("**SV** = EV - PV. SV > 0 indica que se ha ganado más valor del previsto (adelantado); SV < 0 indica retraso.")
        st.metric("SV Final (€)", f"{sv_final:,.2f}")
        if sv_final > 0:
            st.info("Adelantado (EV > PV).")
        elif sv_final < 0:
            st.warning("Retrasado (EV < PV).")
        else:
            st.success("Exactamente en plan.")

    with col_c2:
        with st.expander("❓ ¿Cómo se calcula CV (Cost Variance)?"):
            st.markdown("**CV** = EV - AC. CV > 0 indica que el valor ganado supera lo gastado (ahorro); CV < 0 indica sobrecoste.")
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
                "**EAC** = AC + (BAC - EV) / CPI. Es la estimación del coste total "
                "si el rendimiento futuro (CPI) sigue igual que hasta ahora."
            )
        st.metric("EAC (€)", f"{eac:,.2f}")
    with col_d2:
        with st.expander("❓ ¿Cómo se calcula ETC?"):
            st.markdown("**ETC** = EAC - AC. Es lo que falta por gastar (estimado) para completar el proyecto.")
        st.metric("ETC (€)", f"{etc:,.2f}")
    with col_d3:
        with st.expander("❓ ¿Cómo se calcula VAC?"):
            st.markdown("**VAC** = BAC - EAC. Es la diferencia estimada al finalizar: positivo = ahorro, negativo = sobrecoste estimado.")
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
                "**TCPI** = (BAC - EV) / (BAC - AC). Índice de desempeño que debe mantenerse "
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




def _modo_evm_por_actividad() -> None:
    """Calcula EVM por actividades (ejercicio del PDF) — completo como en el ejercicio resuelto."""
    st.markdown("---")
    st.markdown("### ⚙️ Parámetros del Ejercicio")
    st.caption(
        "Según el calendario de ejecución, al finalizar la primera semana laboral "
        "(viernes 10 de julio 2026): A y B terminadas, C iniciada al 40%. "
        "AC real reportado = 7300 € (sobrecoste importante)."
    )

    bac_act = st.number_input("BAC (Budget at Completion) €", min_value=0.0, value=10000.0, step=500.0, format="%.0f", key="act_bac")
    ac_act = st.number_input("AC (Actual Cost) €", min_value=0.0, value=7300.0, step=100.0, format="%.0f", key="act_ac")

    datos_iniciales_act = {
        "Actividad": ["A", "B", "C", "D", "E"],
        "% Presupuesto": [10.0, 20.0, 30.0, 20.0, 20.0],
        "% Real Completado": [100.0, 100.0, 30.0, 0.0, 0.0],
    }
    df_inicial_act = pd.DataFrame(datos_iniciales_act)

    st.markdown("### 📋 Actividades (Edita % Real si cambia)")
    df_editado_act = st.data_editor(
        df_inicial_act,
        column_config={
            "Actividad": st.column_config.TextColumn("Actividad", disabled=True),
            "% Presupuesto": st.column_config.NumberColumn("% Presupuesto", min_value=0.0, max_value=100.0, step=1.0, format="%.1f"),
            "% Real Completado": st.column_config.NumberColumn("% Real Completado", min_value=0.0, max_value=100.0, step=1.0, format="%.1f"),
        },
        hide_index=True, width="stretch", key="evm__tabla_act",
    )

    st.markdown("### 📅 Estado del Plan al Punto de Control")
    col_plan_a, col_plan_b, col_plan_c = st.columns(3)
    with col_plan_a:
        plan_a = st.number_input("A completado según plan (%)", min_value=0.0, max_value=100.0, value=100.0, step=1.0, key="evm_plan_a")
    with col_plan_b:
        plan_b = st.number_input("B completado según plan (%)", min_value=0.0, max_value=100.0, value=100.0, step=1.0, key="evm_plan_b")
    with col_plan_c:
        plan_c = st.number_input("C completado según plan (%)", min_value=0.0, max_value=100.0, value=40.0, step=1.0, key="evm_plan_c", help="Según calendario: 2/5 de C = 40% planificado")

    actividades = []
    pv_total_act = 0.0
    ev_total_act = 0.0
    ac_total_act = float(ac_act)
    plan_vals = {"A": float(plan_a), "B": float(plan_b), "C": float(plan_c), "D": 0.0, "E": 0.0}

    for _, row in df_editado_act.iterrows():
        act = str(row["Actividad"])
        pct = float(row["% Presupuesto"])
        real = float(row["% Real Completado"])
        cost_plan = float(bac_act) * pct / 100.0
        plan_pct = plan_vals.get(act, 0.0)
        pv = cost_plan * plan_pct / 100.0
        ev = cost_plan * real / 100.0
        actividades.append({
            "Actividad": act,
            "Coste Planificado (€)": cost_plan,
            "% Plan al control": plan_pct,
            "PV (€)": pv,
            "EV (€)": ev,
            "% Real": real,
        })
        pv_total_act += pv
        ev_total_act += ev

    cv_final_act = ev_total_act - ac_total_act
    sv_final_act = ev_total_act - pv_total_act
    spi_final_act = ev_total_act / pv_total_act if pv_total_act > 0 else 0.0
    cpi_final_act = ev_total_act / ac_total_act if ac_total_act > 0 else 0.0
    eac_act = ac_total_act + (float(bac_act) - ev_total_act) / cpi_final_act if cpi_final_act > 0 else 0.0
    etc_act = eac_act - ac_total_act
    vac_act = float(bac_act) - eac_act
    tcpi_act = (float(bac_act) - ev_total_act) / (float(bac_act) - ac_total_act) if (float(bac_act) - ac_total_act) != 0 else 0.0

    st.markdown("---")
    st.subheader("1. Variables Básicas de EVM")
    col_a1, col_a2, col_a3, col_a4 = st.columns(4)
    with col_a1:
        with st.expander("❓ ¿Cómo se calcula BAC?"):
            st.markdown(f"**BAC** = Presupuesto total aprobado = {float(bac_act):,.0f} €.")
        st.metric("BAC (€)", f"{float(bac_act):,.2f}")
    with col_a2:
        with st.expander("❓ ¿Cómo se calcula PV?"):
            st.markdown(f"**PV** = suma según calendario (A={plan_a}%, B={plan_b}%, C={plan_c}%): {pv_total_act:,.0f} €.")
        st.metric("PV (€)", f"{pv_total_act:,.2f}")
    with col_a3:
        with st.expander("❓ ¿Cómo se calcula EV?"):
            st.markdown(f"**EV** = suma del valor real = {ev_total_act:,.0f} € (A y B completas, C según % real editado).")
        st.metric("EV (€)", f"{ev_total_act:,.2f}")
    with col_a4:
        with st.expander("❓ ¿Cómo se calcula AC?"):
            st.markdown(f"**AC** = Coste real acumulado = {float(ac_total_act):,.0f} €.")
        st.metric("AC (€)", f"{float(ac_total_act):,.2f}")

    st.markdown("---")
    st.subheader("2. Métricas EVM (como en el ejercicio resuelto)")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        with st.expander("❓ Fórmulas CV y SV"):
            st.markdown(
                f"**CV = EV - AC = {ev_total_act:,.0f} - {ac_total_act:,.0f} = {cv_final_act:,.0f} EUR** (sobrecoste si negativo). "
                f"**SV = EV - PV = {ev_total_act:,.0f} - {pv_total_act:,.0f} = {sv_final_act:,.0f} EUR** (retraso si negativo)."
            )
        st.metric("CV (€)", f"{cv_final_act:,.2f}", delta="Sobrecoste importante" if cv_final_act < 0 else "Ahorro")
        st.metric("SV (€)", f"{sv_final_act:,.2f}", delta="Retraso" if sv_final_act < 0 else "Adelantado")
    with col_b2:
        with st.expander("❓ Fórmulas CPI y SPI"):
            st.markdown(
                f"**CPI = EV / AC = {ev_total_act:,.0f} / {ac_total_act:,.0f} = {cpi_final_act:.2f}** (baja eficiencia si <1). "
                f"**SPI = EV / PV = {ev_total_act:,.0f} / {pv_total_act:,.0f} = {spi_final_act:.2f}** (retraso si <1)."
            )
        st.metric("CPI", f"{cpi_final_act:.2f}")
        st.metric("SPI", f"{spi_final_act:.2f}")

    st.markdown("---")
    st.subheader("3. Proyección de Coste")
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    with col_d1:
        with st.expander("❓ Fórmula EAC"):
            st.markdown(f"EAC = BAC / CPI = {bac_act:,.0f} / {cpi_final_act:.2f} = {eac_act:,.0f} EUR")
        st.metric("EAC (€)", f"{eac_act:,.2f}", delta=f"{vac_act:,.0f} € vs BAC" if vac_act != 0 else None)
    with col_d2:
        with st.expander("❓ Fórmula ETC"):
            st.markdown(f"ETC = EAC - AC = {eac_act:,.0f} - {ac_total_act:,.0f} = {etc_act:,.0f} EUR (lo que falta por gastar estimado).")
        st.metric("ETC (€)", f"{etc_act:,.2f}")
    with col_d3:
        with st.expander("❓ Fórmula TCPI(BAC)"):
            st.markdown(f"TCPI(BAC) = ({bac_act:,.0f} - {ev_total_act:,.0f}) / ({bac_act:,.0f} - {ac_total_act:,.0f}) = {tcpi_act:.2f}. Para cerrar en BAC hay que duplicar eficiencia.")
        st.metric("TCPI(BAC)", f"{tcpi_act:.2f}")
    with col_d4:
        with st.expander("❓ Fórmula VAC"):
            st.markdown(f"VAC = BAC - EAC = {bac_act:,.0f} - {eac_act:,.0f} = {vac_act:,.0f} EUR. Se espera terminar por encima si VAC < 0.")
        st.metric("VAC (€)", f"{vac_act:,.2f}")

    st.markdown("---")
    st.subheader("4. Conclusiones")
    st.markdown(
        f"Proyecto ligeramente retrasado (SPI = {spi_final_act:.2f}) pero con sobrecoste muy alto (CPI = {cpi_final_act:.2f}). Si nada cambia, coste final ~{eac_act:,.0f} EUR. Para {bac_act:,.0f} EUR hay que duplicar eficiencia (TCPI = {tcpi_act:.2f}): poco realista -> replanificar."
    )

    st.markdown("---")
    st.subheader("5. Desglose por Actividad")
    df_actividades = pd.DataFrame(actividades)
    st.dataframe(
        df_actividades.style.format({
            "Coste Planificado (€)": "{:.2f} €",
            "PV (€)": "{:.2f} €",
            "EV (€)": "{:.2f} €",
            "% Plan al control": "{:.1f} %",
            "% Real": "{:.1f} %",
        }),
        hide_index=True, width="stretch",
    )


def _modo_evm() -> None:
    "Ventana de EVM."
    st.markdown("### 📊 Gestión del Valor Ganado (EVM)")
    with st.expander("❓ Conceptos Básicos...", expanded=True):
        st.markdown("- BAC... - PV... - EV... - AC... - SV... - CV... - SPI... - CPI...")
    modo_evm = st.selectbox("Modo de cálculo:", ["Modo estándar", "Modo por actividad"], index=0, key="evm__modo")
    if modo_evm == "Modo estándar":
        # El bloque estándar está en la función auxiliar
        _modo_evm_estandar()
    else:
        # El bloque por actividad está en la función auxiliar
        _modo_evm_por_actividad()
# ---------------------------------------------------------------------------
# Punto de entrada de la vista (modificado para incluir EVM)
# ---------------------------------------------------------------------------
def render(_ctx: dict) -> None:
    "Dibuja el Planificador con sus dos modos."
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
