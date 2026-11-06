"""Aplicacion unificada: las tres herramientas de ingenieria en un solo enlace.

Este archivo es el punto de entrada. Su responsabilidad es deliberadamente
pequena:

1. Configurar la pagina UNA vez (Streamlit solo admite una configuracion por
   app; antes cada una de las tres apps tenia la suya).
2. Dibujar la configuracion de IA UNA vez en la barra lateral, de modo que la
   API Key se introduce una sola vez y sirve para todas las pestanas que la
   necesiten.
3. Repartir las tres herramientas en tres pestanas de primer nivel.

Cada herramienta conserva su logica y su interfaz; lo unico que cambia es que
ahora viven en la misma pagina.

Ejecucion
---------
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

import config_ia
from views import diseno_producto, planificador, viabilidad

# Unica llamada a set_page_config de toda la aplicacion.
st.set_page_config(
    page_title="Herramientas de Ingeniería",
    page_icon="🛠️",
    layout="wide",
)

st.title("🛠️ Herramientas de Ingeniería para Proyectos")
st.caption(
    "Planificación, viabilidad económica y diseño de producto en un único enlace."
)

# Configuracion de IA en la barra lateral: se introduce la API Key una vez y
# queda disponible para el resto de la aplicacion.
ctx = config_ia.render_sidebar_ia()

# Pestanas de primer nivel, perezosas: solo se ejecuta la que esta visible.
tab_diseno, tab_plan, tab_viab = st.tabs(
    ["🛠️ Diseño de Producto", "📊 Planificador", "💰 Viabilidad"],
    on_change="rerun",
    key="nav_principal",
)

if tab_diseno.open:
    with tab_diseno:
        diseno_producto.render(ctx)

if tab_plan.open:
    with tab_plan:
        planificador.render(ctx)

if tab_viab.open:
    with tab_viab:
        viabilidad.render(ctx)