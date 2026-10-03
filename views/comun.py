"""Utilidades compartidas por las vistas de la aplicacion unificada.

El problema que resuelven
-------------------------
Las tres apps originales mostraban sus resultados dentro de un
`if st.button(...):`. En Streamlit, un boton solo devuelve `True` en la
ejecucion inmediatamente posterior al clic, asi que al tocar cualquier widget
de otra pestana el resultado desaparecia.

En una pagina unificada eso pasamuchisimo: el usuario pulsa "Calcular" en
AMFE, va a Viabilidad a cambiar un flujo de caja y al volver a AMFE el
resultado ya no esta.

`guardar_resultado` / `leer_resultado` / `hay_resultado` persistsen el
resultado en `st.session_state`, de modo que sobrevive a los reruns caused por
interacciones en otras pestanas.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st


def guardar_resultado(clave: str, valor) -> None:
    """Guarda un resultado de forma persistente entre ejecuciones."""
    st.session_state[clave] = valor


def leer_resultado(clave: str, por_defecto=None):
    """Lee un resultado guardado. Devuelve `por_defecto` si no hay ninguno."""
    return st.session_state.get(clave, por_defecto)


def hay_resultado(clave: str) -> bool:
    """Indica si existe un resultado guardado para esa clave."""
    return clave in st.session_state


def borrar_resultado(clave: str) -> None:
    """Elimina un resultado guardado."""
    st.session_state.pop(clave, None)


def boton_calcular(texto: str, clave: str, resultado: str) -> bool:
    """Dibuja el boton de calculo y un boton para limpiar el resultado.

    Returns
    -------
    bool
        `True` si en esta ejecucion el usuario ha pulsado "Calcular".
        `False` en caso contrario (tambien cuando acaba de pulsar "Limpiar").

    El boton de limpiar solo aparece cuando hay un resultado guardado, y al
    pulsarlo borra el resultado y relanza la app para que el estado de los
    widgets quede limpio.
    """
    columnas = st.columns([3, 1])
    with columnas[0]:
        pulsar = st.button(texto, key=f"{clave}__calcular", width="stretch")
    with columnas[1]:
        if hay_resultado(resultado):
            if st.button("🗑️ Limpiar", key=f"{clave}__limpiar", width="stretch"):
                borrar_resultado(resultado)
                st.rerun()
    return pulsar


def mostrar_figura(fig) -> None:
    """Pinta una figura de matplotlib y la cierra para no acumular memoria.

    Sin este `close`, Streamlit avisa a partir de unas 20 figuras abiertas en
    la misma sesion, algo facil de alcanzar al combinar las tres apps.
    """
    st.pyplot(fig, width="stretch")
    plt.close(fig)