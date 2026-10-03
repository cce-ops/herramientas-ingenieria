"""Configuracion del proveedor de IA de la aplicacion unificada.

Este modulo concentra TODA la configuracion de IA en un unico sitio:

* El selector de proveedor (solo Google Gemini) y de modelo.
* El campo de API Key, que se rellena **una sola vez** en la barra lateral y
  queda disponible para cualquier pestana que necesite IA (PDS e Ishikawa).
* La funcion `evaluar_texto_llm()`, unica funcion de la aplicacion que hace
  llamadas a la red.

Reintentos y modelos de respaldo
--------------------------------
Si la llamada a un modelo falla, `evaluar_texto_llm` reintenta el mismo modelo
esperando unos segundos entre intento e intento. Si se agotan los reintentos,
pasa al siguiente modelo de la lista y repite el proceso, empezando por el
modelo que el usuario ha seleccionado y siguiendo el orden de la lista. Si se
llega al final, vuelve a empezar por el principio.

Los errores que no tienen sentido reintentar (una API Key mal escrita, un
modelo inexistente) cortan la cadena inmediatamente: reintentar 8 modelos con
una clave invalida solo Hareia esperar al usuario sin resultado.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

# ---------------------------------------------------------------------------
# Parametros del reintento
# ---------------------------------------------------------------------------
# Espera entre reintentos del mismo modelo, en segundos.
ESPERA_REINTENTO_SEG = 3.0

# Intentos por modelo antes de pasar al siguiente.
MAX_INTENTOS_POR_MODELO = 3

# Codigos de error que no tienen sentido reintentar ni repetir en otro modelo:
# la peticion es invalida por si sola, pruebes lo que pruebes.
#   400 peticion incorrecta · 401 clave invalida · 403 sin permiso
#   404 modelo inexistente · 403/404 también cubren el caso de un nombre de
#   modelo mal escrito en la lista.
CODIGOS_SIN_REINTENTO = {400, 401, 403, 404}


# ---------------------------------------------------------------------------
# Catalogo de proveedores
# ---------------------------------------------------------------------------
PROVEEDORES: dict[str, dict] = {
    "Google Gemini": {
        "modelos": {
            "Gemini 3.8 Flash (Más inteligente, flujos complejos)": "gemini-3.8-flash",
            "Gemini 3.8 Live (Voz, baja latencia)": "gemini-3.8-live",
            "Gemini 3.8 Live Extended Thinking (Alto razonamiento)": (
                "gemini-3.8-live-extended-thinking"
            ),
            "Gemini 3.7 Flash (Programación y varios pasos)": "gemini-3.7-flash",
            "Gemini 3.6 Flash (Equilibrio tareas cotidianas)": "gemini-3.6-flash",
            "Gemini 3.5 Flash (Velocidad para cargas rutinarias)": "gemini-3.5-flash",
            "Gemini 3.5 Flash-Lite (Más rápido y rentable)": "gemini-3.5-flash-lite",
            "Gemini 3.1 Flash-Lite (Rendimiento Frontier)": "gemini-3.1-flash-lite",
        },
        "ayuda": "Obtén tu clave en Google AI Studio",
    },
}

# Clave de session_state donde se guarda la clave de cada proveedor.
_CLAVES = "claves_ia"


@dataclass
class RespuestaIA:
    """Resultado de una llamada a la IA, con la trazabilidad del reintento.

    `texto` es lo que se muestra al usuario, sea la respuesta del modelo o el
    aviso de error. El resto de campos sirven para poder explicar en pantalla
    qué modelo respondió de verdad, porque con el mecanismo de respaldo puede
    no ser el que el usuario eligió.
    """

    texto: str
    ok: bool
    modelo: str = ""
    intentos: int = 0
    hubo_fallback: bool = False
    descartados: list[str] = field(default_factory=list)


def _clave_de_widget(proveedor: str) -> str:
    """Nombre estable de la clave del widget para un proveedor dado."""
    return f"api_key_{proveedor.lower().replace(' ', '_')}"


def render_sidebar_ia() -> dict:
    """Dibuja la configuracion de IA en la barra lateral.

    Se llama UNA vez desde `app.py`, antes de las pestanas. Devuelve el
    contexto de IA que consumen las vistas.

    Returns
    -------
    dict con `proveedor`, `api_key`, `model_name`, `etiqueta_modelo` y
    `modelos_disponibles` (la lista de modelos, para la cadena de
    respaldo).
    """
    import streamlit as st

    st.sidebar.header("⚙️ Configuración de IA")
    st.sidebar.caption(
        "Introduce tu clave **una sola vez**. Queda disponible para toda la "
        "aplicación."
    )

    # Un solo punto de entrada para las claves de todos los proveedores.
    if _CLAVES not in st.session_state:
        st.session_state[_CLAVES] = {}

    proveedor = st.sidebar.selectbox(
        "Proveedor de IA:",
        list(PROVEEDORES.keys()),
        key="sel_proveedor_ia",
    )
    modelos = PROVEEDORES[proveedor]["modelos"]

    etiqueta_modelo = st.sidebar.selectbox(
        "Modelo:",
        list(modelos.keys()),
        key=f"sel_modelo_{proveedor}",
    )

    api_key = st.sidebar.text_input(
        f"{proveedor} API Key",
        type="password",
        value=st.session_state[_CLAVES].get(proveedor, ""),
        key=_clave_de_widget(proveedor),
        help=PROVEEDORES[proveedor]["ayuda"],
    )
    # Guardamos la clave introducida para recuperarla al volver a este proveedor.
    st.session_state[_CLAVES][proveedor] = api_key

    return {
        "proveedor": proveedor,
        "api_key": api_key,
        "model_name": modelos[etiqueta_modelo],
        "etiqueta_modelo": etiqueta_modelo,
        "modelos_disponibles": list(modelos.values()),
    }


def hay_api_key(ctx: dict) -> bool:
    """Indica si hay una clave utilizable en el contexto de IA."""
    return bool(ctx.get("api_key"))


def nombre_modelo_corto(ctx: dict) -> str:
    """Nombre del modelo sin el texto entre parentesis, para los spinners."""
    return ctx["etiqueta_modelo"].split("(")[0].strip()


def _cadena_de_respaldo(ctx: dict) -> list[str]:
    """Modelos a probar, empezando por el elegido y siguiendo el orden.

    Al llegar al último vuelve a empezar por el primero, para que el modelo
    seleccionado siempre tenga alternativas aunque se elija el final de la lista.
    """
    modelos = ctx["modelos_disponibles"]
    elegido = ctx["model_name"]
    if elegido in modelos:
        inicio = modelos.index(elegido)
        return modelos[inicio:] + modelos[:inicio]
    return list(modelos)


def _es_error_fatal(excepcion: Exception) -> bool:
    """Indica si el error no mejora reintentando ni cambiando de modelo."""
    if isinstance(excepcion, genai_errors.APIError):
        return excepcion.code in CODIGOS_SIN_REINTENTO
    return False


def _llamar_gemini(api_key: str, modelo: str, prompt_sistema: str, texto: str) -> str:
    """Una llamada al modelo. Si va bien, devuelve el texto."""
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=modelo,
        contents=texto,
        config=genai_types.GenerateContentConfig(
            system_instruction=prompt_sistema,
            temperature=0.3,
        ),
    )
    return response.text


def evaluar_texto_llm(
    ctx: dict,
    prompt_sistema: str,
    texto_usuario: str,
    max_intentos: int = MAX_INTENTOS_POR_MODELO,
    espera: float = ESPERA_REINTENTO_SEG,
    on_progreso=None,
) -> RespuestaIA:
    """Evalua un texto con IA, reintentando y cambiando de modelo si hace falta.

    Recorre la cadena de modelos empezando por el seleccionado. Cada modelo se
    intenta hasta `max_intentos` veces, esperando `espera` segundos entre
    intentos. Al agotar un modelo pasa al siguiente. Los errores no
    reintentables cortan la cadena.

    Parameters
    ----------
    ctx : dict
        Contexto devuelto por `render_sidebar_ia`.
    prompt_sistema : str
        Instruccion de rol para el modelo.
    texto_usuario : str
        Contenido a evaluar.
    max_intentos : int
        Intentos por modelo.
    espera : float
        Segundos de espera entre reintentos.
    on_progreso : callable, opcional
        Se llama como `on_progreso(modelo, intento, max_intentos)` antes de
        cada intento. Sirve para ir Actualizando el texto del spinner.

    Returns
    -------
    RespuestaIA
    """
    if not hay_api_key(ctx):
        return RespuestaIA(
            texto=(
                f"⚠️ Introduce tu API Key de {ctx['proveedor']} en la barra "
                "lateral para usar esta función."
            ),
            ok=False,
        )

    descartados: list[str] = []
    ultimo_error = ""

    for modelo in _cadena_de_respaldo(ctx):
        for intento in range(1, max_intentos + 1):
            if on_progreso is not None:
                on_progreso(modelo, intento, max_intentos)
            try:
                texto = _llamar_gemini(
                    ctx["api_key"], modelo, prompt_sistema, texto_usuario
                )
            except Exception as exc:  # noqa: BLE001 - se clasifica abajo
                ultimo_error = f"{type(exc).__name__}: {exc}"

                if _es_error_fatal(exc):
                    return RespuestaIA(
                        texto=(
                            f"⚠️ {ctx['proveedor']} ha rechazado la petición y no "
                            f"tiene sentido reintentar.\n\n"
                            f"**Motivo:** {ultimo_error}\n\n"
                            "Revisa que la API Key sea correcta y que el nombre "
                            "del modelo exista."
                        ),
                        ok=False,
                        modelo=modelo,
                        intentos=intento,
                        descartados=descartados,
                    )

                # No es el ultimo intento: se espera y se repite.
                if intento < max_intentos:
                    time.sleep(espera)
                continue

            # Camino feliz: este modelo ha respondido.
            return RespuestaIA(
                texto=texto,
                ok=True,
                modelo=modelo,
                intentos=intento,
                hubo_fallback=bool(descartados),
                descartados=list(descartados),
            )

        # Se han agotado los intentos de este modelo: al siguiente.
        descartados.append(modelo)

    return RespuestaIA(
        texto=(
            f"⚠️ No se ha podido obtener respuesta de {ctx['proveedor']}.\n\n"
            f"**Modelos probados:** {len(descartados)}\n"
            f"**Último error:** {ultimo_error}"
        ),
        ok=False,
        intentos=max_intentos,
        hubo_fallback=len(descartados) > 1,
        descartados=descartados,
    )


def resumen_del_intento(respuesta: RespuestaIA, modelo_elegido: str) -> str | None:
    """Texto para avisar de reintentos o de cambio de modelo, o None si no hace falta.

    Es importante ser transparente: con el mecanismo de respaldo, el usuario
    puede recibir la respuesta de un modelo distinto del que eligió sin
    enterarse.
    """
    if not respuesta.ok:
        return None

    partes = []
    if respuesta.modelo != modelo_elegido:
        descartados = ", ".join(f"`{m}`" for m in respuesta.descartados)
        partes.append(
            f"⚠️ El modelo elegido (`{modelo_elegido}`) no respondió. Se ha usado "
            f"`{respuesta.modelo}` como respaldo."
            + (f" Modelos descartados: {descartados}." if descartados else "")
        )
    if respuesta.intentos > 1:
        partes.append(
            f"Respondió tras {respuesta.intentos} intentos "
            f"(el primero falló y se reintentó)."
        )
    return " ".join(partes) if partes else None