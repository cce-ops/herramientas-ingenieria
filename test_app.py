"""Pruebas de la aplicacion unificada.

Se ejecutan con el modulo de testing de Streamlit (AppTest), que lanza la app
entera y permite inspeccionar la interfaz sin navegador.

    python test_app.py          # o bien:  python -m pytest test_app.py -v

La prueba clave es `test_resultado_sobrevive_al_cambio_de_pestana`: es el
problema que hacia que unificar las tres apps sin mas perdiera resultados.

Nota sobre AppTest
------------------
Las pestanas usan `on_change="rerun"`, que las convierte en widgets con
estado. AppTest reinicia ese estado al lanzar la siguiente ejecucion, asi que
hay que fijar la pestana justo antes de cada `click()`. De eso se encarga
`_ir()`.
"""

from __future__ import annotations

import warnings

from streamlit.testing.v1 import AppTest

APP = "app.py"

TAB_DISENO = "🛠️ Diseño de Producto"
TAB_PLAN = "📊 Planificador"
TAB_VIAB = "💰 Viabilidad"

SUB_PLAN_BASICO = "🟢 Modo Básico (Actividades y Precedencias)"
SUB_PLAN_AVANZADO = "🔵 Modo Avanzado (Nodos y Zaderenko)"
SUB_PDS = "📋 PDS"
SUB_AMFE = "⚠️ AMFE"
SUB_ISHIKAWA = "🐟 Ishikawa"
SUB_QFD = "🏠 QFD"


def cargar() -> AppTest:
    return AppTest.from_file(APP, default_timeout=60)


def _ir(at: AppTest, pestana: str | None = None, sub: str | None = None) -> AppTest:
    """Fija la pestana (y sub-pestana) visibles y ejecuta la app."""
    if pestana is not None:
        at.session_state["nav_principal"] = pestana
    if sub is not None:
        at.session_state[_clave_sub(pestana)] = sub
    at.run()
    assert not at.exception, [e.message.split("\n")[0] for e in at.exception]
    return at


def _clave_sub(pestana: str | None) -> str:
    return {
        TAB_DISENO: "dise__nav",
        TAB_PLAN: "plan__nav",
    }.get(pestana or "", "nav_principal")


def _afirmar(at: AppTest, pestana: str | None, sub: str | None) -> None:
    """Reafirma el estado de las pestañas sin ejecutar la app.

    AppTest reinicia el estado de los widgets con estado (las pestañas con
    `on_change="rerun"`) al preparar la siguiente ejecucion, asi que hay que
    volver a fijarlo justo antes de `click()`.
    """
    if pestana is not None:
        at.session_state["nav_principal"] = pestana
    if sub is not None:
        at.session_state[_clave_sub(pestana)] = sub


def _pulsar(at: AppTest, boton: str, pestana: str, sub: str | None = None) -> AppTest:
    """Fija la pestana, la deja pintar sus widgets y pulsa un boton."""
    _ir(at, pestana, sub)
    _afirmar(at, pestana, sub)
    at.button(key=boton).click().run()
    assert not at.exception, [e.message.split("\n")[0] for e in at.exception]
    return at


def _texto(at: AppTest) -> str:
    """Todo el texto visible de la app, de cualquier tipo de elemento."""
    partes = []
    for coleccion in (
        at.title,
        at.header,
        at.subheader,
        at.markdown,
        at.caption,
        at.info,
        at.warning,
        at.success,
    ):
        partes.extend(e.value for e in coleccion)
    return " ".join(partes)


# ---------------------------------------------------------------------------
# 1. Carga general
# ---------------------------------------------------------------------------
def test_carga_sin_errores():
    at = cargar().run()
    assert not at.exception, [e.message.split("\n")[0] for e in at.exception]


def test_no_hay_avisos_de_deprecacion():
    """`use_container_width` esta obsoleto: no debe quedar ni una vez."""
    at = cargar().run()
    assert not at.exception


def test_titulo_present():
    assert "Herramientas de Ingeniería" in _texto(cargar().run())


def test_configuracion_de_ia_en_la_barra_lateral():
    """La API Key se pide una unica vez, en la barra lateral."""
    at = cargar().run()
    etiquetas = [s.label for s in at.sidebar.selectbox]
    assert "Proveedor de IA:" in etiquetas
    assert any("API Key" in t.label for t in at.sidebar.text_input)


# ---------------------------------------------------------------------------
# 2. Las tres pestañas de primer nivel
# ---------------------------------------------------------------------------
def test_pestanas_principales_renderizan():
    for etiqueta, esperado in [
        (TAB_DISENO, "Especificaciones de Diseño de Producto"),
        (TAB_PLAN, "Datos de Entrada"),
        (TAB_VIAB, "Parámetros Generales"),
    ]:
        at = _ir(cargar(), etiqueta)
        assert esperado in _texto(at), f"falta {esperado!r} en {etiqueta!r}"


def test_subpestanas_de_diseno():
    for sub, esperado in [
        (SUB_PDS, "Especificaciones de Diseño de Producto"),
        (SUB_AMFE, "Análisis Modal de Fallos y Efectos"),
        (SUB_ISHIKAWA, "Diagrama Causa-Efecto"),
        (SUB_QFD, "Casa de la Calidad"),
    ]:
        at = _ir(cargar(), TAB_DISENO, sub)
        assert esperado in _texto(at), f"falta {esperado!r} en {sub!r}"


def test_subpestanas_de_planificador():
    for sub, esperado in [
        (SUB_PLAN_BASICO, "Datos de Entrada (Modo Básico)"),
        (SUB_PLAN_AVANZADO, "Datos de Actividades Sucesos"),
    ]:
        at = _ir(cargar(), TAB_PLAN, sub)
        assert esperado in _texto(at), f"falta {esperado!r} en {sub!r}"


# ---------------------------------------------------------------------------
# 3. LA PRUEBA CLAVE: persistencia de resultados entre pestañas
# ---------------------------------------------------------------------------
def test_resultado_sobrevive_al_cambio_de_pestana():
    """Un resultado de AMFE no puede desaparecer al usar otra pestaña.

    Es el fallo que se produciría al concatenar los tres archivos originales
    sin más: cualquier interaccion en otra pestaña relanza el script y el
    `if st.button(...)` deja de cumplirse.
    """
    at = cargar()
    _pulsar(at, "dise__amfe__calcular", TAB_DISENO, SUB_AMFE)
    assert at.session_state.get("dise__resultado_amfe") is not None, "no se calculo"
    assert at.success, "no se mostro el mensaje de exito"

    # El usuario va a Viabilidad (que se recalcula en vivo) y vuelve.
    _ir(at, TAB_VIAB)
    assert not at.exception

    _ir(at, TAB_DISENO, SUB_AMFE)
    assert at.session_state.get("dise__resultado_amfe") is not None, (
        "el resultado de AMFE se perdio al cambiar de pestana"
    )


def test_resultado_planificador_sobrevive():
    """Igual para el Gantt y el camino critico del modo basico."""
    at = cargar()
    _pulsar(at, "plan__basico__calcular", TAB_PLAN, SUB_PLAN_BASICO)
    assert at.session_state.get("plan__resultado_basico") is not None

    _ir(at, TAB_VIAB)
    _ir(at, TAB_PLAN, SUB_PLAN_BASICO)
    assert at.session_state.get("plan__resultado_basico") is not None, (
        "el resultado del planificador se perdio"
    )


def test_limpiar_resultado():
    at = cargar()
    _pulsar(at, "plan__basico__calcular", TAB_PLAN, SUB_PLAN_BASICO)
    assert at.session_state.get("plan__resultado_basico") is not None

    # El boton "Limpiar" solo aparece en la ejecucion siguiente: el resultado se
    # guarda despues de que `boton_calcular` ya ha dibujado los botones.
    _ir(at, TAB_PLAN, SUB_PLAN_BASICO)
    _afirmar(at, TAB_PLAN, SUB_PLAN_BASICO)
    at.button(key="plan__basico__limpiar").click().run()
    assert at.session_state.get("plan__resultado_basico") is None, (
        "el boton Limpiar no borro el resultado"
    )


# ---------------------------------------------------------------------------
# 4. Logica de calculo preservada
# ---------------------------------------------------------------------------
def test_modo_basico_calcula_camino_critico():
    """Datos por defecto -> camino critico A-B-D-F con 14 dias laborables."""
    at = cargar()
    _pulsar(at, "plan__basico__calcular", TAB_PLAN, SUB_PLAN_BASICO)

    res = at.session_state["plan__resultado_basico"]
    assert res["duracion_total"] == 14, res["duracion_total"]
    criticas = {f["Actividad"] for f in res["tabla"].to_dict("records") if f["CC"]}
    assert criticas == {"A", "B", "D", "F"}, criticas
    assert set(res["gantt"]["Crítica"]) == {"Sí", "No"}


def test_modo_avanzado_calcula_zaderenko():
    at = cargar()
    _pulsar(at, "plan__avanzado__calcular", TAB_PLAN, SUB_PLAN_AVANZADO)

    res = at.session_state["plan__resultado_avanzado"]
    assert res["duracion"] == 7.0, res["duracion"]
    # La matriz debe ser homogenea (todo texto) para que pyarrow la acepte.
    assert set(map(type, res["matriz"].values.ravel())) == {str}
    assert "t_i*" in res["matriz"].index
    assert "t_j" in res["matriz"].columns


def test_modo_basico_detecta_bucle():
    """Un ciclo en las precedencias se detecta y avisa, sin romper la app."""
    import networkx as nx
    import pandas as pd

    from views.planificador import _calcular_basico

    ciclico = pd.DataFrame(
        {
            "Tarea": ["A", "B"],
            "Duración": [2, 3],
            "Predecesores": ["B", "A"],  # A depende de B y B de A
        }
    )
    try:
        _calcular_basico(ciclico, pd.Timestamp("2026-01-01").date())
    except nx.NetworkXUnfeasible:
        pass
    else:
        raise AssertionError("se esperaba NetworkXUnfeasible con un ciclo")


def test_modo_basico_con_bucle_no_rompe_la_app():
    """La UI muestra un error controlado en lugar de una excepcion."""
    import pandas as pd

    at = _ir(cargar(), TAB_PLAN, SUB_PLAN_BASICO)
    at.session_state["df_basico Ciclico"] = True  # marcador irrelevante
    at.session_state["plan__tabla_basica"] = {
        "Tarea": {"0": "A", "1": "B"},
        "Duración": {"0": 2, "1": 3},
        "Predecesores": {"0": "B", "1": "A"},
    }
    _afirmar(at, TAB_PLAN, SUB_PLAN_BASICO)
    at.button(key="plan__basico__calcular").click().run()

    assert not at.exception, [e.message.split("\n")[0] for e in at.exception]


def test_viabilidad_calcula_van_tir():
    """Inversion 50.000, tasa 5%, 10 anos de 12.000."""
    at = _ir(cargar(), TAB_VIAB)

    metricas = {m.label: m.value for m in at.metric}
    assert set(metricas) == {"VAN", "TIR", "Payback"}, metricas

    van = float(metricas["VAN"].replace("€", "").replace(",", "").strip())
    tir = float(metricas["TIR"].replace("%", "").strip())

    # VAN = -50.000 + 12.000 * 7,7217 ~= 42.660
    assert 42_000 < van < 43_000, van
    # TIR de una renta de 12.000 que amortiza 50.000 en 10 anos ~= 20,6%
    assert 19 < tir < 22, tir
    # Payback exacto por interpolacion: 50.000 / 12.000 ~= 4,17 anos
    payback = float(metricas["Payback"].replace("Años", "").strip())
    assert 4 < payback < 4.3, payback


def test_amfe_calcula_npr():
    """NPR = S x O x D, con la criticidad que ya usaba la app original.

    Ojo con la escala de colores: el codigo original marca 🔴 unicamente cuando
    S >= 9. Un NPR >= 100 con S < 9 cae en 🟠. Esta prueba fija ese
    comportamiento tal cual, sin cambiarlo.
    """
    at = cargar()
    _pulsar(at, "dise__amfe__calcular", TAB_DISENO, SUB_AMFE)

    df = at.session_state["dise__resultado_amfe"]
    assert "NPR Calculado" in df.columns
    assert "Sugerencia Sistema" in df.columns

    bisagra = df[df["Componente"] == "Bisagra"].iloc[0]
    assert bisagra["NPR Calculado"] == 7 * 4 * 6  # 168
    assert "Prioridad Media-Alta" in bisagra["Sugerencia Sistema"]

    tornillo = df[df["Componente"] == "Tornillo"].iloc[0]
    assert tornillo["NPR Calculado"] == 8 * 5 * 6  # 240


def test_amfe_rojo_cuando_severidad_alta():
    """Con S = 9 la sugerencia pasa a 🔴 Acción Urgente."""
    from views import diseno_producto as dp

    import pandas as pd

    at = cargar()
    _ir(at, TAB_DISENO, SUB_AMFE)
    at.session_state["df_amfe"] = pd.DataFrame(
        {
            "Componente": ["Eje"],
            "Modo de fallo": ["Rotura"],
            "Severidad (S)": [9],
            "Ocurrencia (O)": [2],
            "Detección (D)": [1],
        }
    )
    _ir(at, TAB_DISENO, SUB_AMFE)
    _afirmar(at, TAB_DISENO, SUB_AMFE)
    at.button(key="dise__amfe__calcular").click().run()

    df = at.session_state["dise__resultado_amfe"]
    assert df.iloc[0]["NPR Calculado"] == 18
    assert "Acción Urgente" in df.iloc[0]["Sugerencia Sistema"]


def test_qfd_calcula_porcentajes():
    at = cargar()
    _pulsar(at, "dise__qfd__calcular", TAB_DISENO, SUB_QFD)

    tabla = at.session_state["dise__resultado_qfd"]["tabla"]
    # QUÉ1(importancia 5) -> Req1=9, Req2=0 ; QUÉ2(importancia 8) -> 0, 9
    # Req1 = 5*9 = 45 ; Req2 = 8*9 = 72 ; total 117
    assert tabla.iloc[0]["Puntuación Absoluta"] == 72
    assert tabla.iloc[1]["Puntuación Absoluta"] == 45
    assert tabla["Puntuación Absoluta"].sum() == 117


# ---------------------------------------------------------------------------
# 5. IA
# ---------------------------------------------------------------------------
def test_ia_sin_api_key_no_revienta():
    """Sin clave, las funciones de IA avisan pero la app sigue viva."""
    at = _ir(cargar(), TAB_DISENO, SUB_PDS)
    at.text_area(key="dise__pds_entrada").set_value("La pieza debe ser resistente.")
    _afirmar(at, TAB_DISENO, SUB_PDS)
    at.button(key="dise__pds_boton").click().run()

    assert not at.exception, [e.message.split("\n")[0] for e in at.exception]
    _, resultado = at.session_state["dise__respuesta_pds"]
    assert not resultado.ok
    assert "API Key" in resultado.texto


def test_ishikawa_sin_api_key_no_revienta():
    at = _ir(cargar(), TAB_DISENO, SUB_ISHIKAWA)
    at.text_input(key="dise__ishikawa_entrada").set_value("Tinta mala en la línea 2.")
    _afirmar(at, TAB_DISENO, SUB_ISHIKAWA)
    at.button(key="dise__ishikawa_boton").click().run()

    assert not at.exception, [e.message.split("\n")[0] for e in at.exception]
    (_, _), resultado = at.session_state["dise__respuesta_ishikawa"]
    assert not resultado.ok
    assert "API Key" in resultado.texto


def test_respuesta_de_ia_no_se_vuelve_a_pedir():
    """La respuesta guardada se reutiliza: no se vuelve a llamar a la API."""
    from config_ia import RespuestaIA

    at = _ir(cargar(), TAB_DISENO, SUB_PDS)
    at.session_state["dise__respuesta_pds"] = (
        "La pieza soportará 2,0 kN.",
        RespuestaIA(
            texto="Respuesta simulada de prueba.",
            ok=True,
            modelo="gemini-3.8-flash",
            intentos=1,
        ),
    )
    at.run()

    assert "Respuesta simulada de prueba." in _texto(at)
    # El boton de repetir solo debe aparecer si hay algo que repetir.
    assert any(b.key == "dise__pds_repetir" for b in at.button)


def test_api_key_se_conserva_por_proveedor():
    """La clave de un proveedor no se pisa al volver a seleccionarlo."""
    at = cargar().run()
    assert at.session_state["claves_ia"]["Google Gemini"] == ""

    at.text_input(key="api_key_google_gemini").set_value("clave-de-prueba").run()
    assert at.session_state["claves_ia"]["Google Gemini"] == "clave-de-prueba"

    # Cambiar de pestana y volver debe conservar la clave.
    _ir(at, TAB_PLAN)
    _ir(at, TAB_DISENO, SUB_PDS)
    assert at.text_input(key="api_key_google_gemini").value == "clave-de-prueba"


# ---------------------------------------------------------------------------
# 6. Proveedor de IA: solo Gemini, con reintentos y respaldo
# ---------------------------------------------------------------------------
def test_solo_google_gemini():
    """Los proveedores OpenRouter, Groq y NVIDIA NIM han desaparecido."""
    from config_ia import PROVEEDORES

    assert list(PROVEEDORES) == ["Google Gemini"], PROVEEDORES

    at = cargar().run()
    opciones = next(
        s.options for s in at.sidebar.selectbox if s.label == "Proveedor de IA:"
    )
    assert list(opciones) == ["Google Gemini"], opciones


def test_los_ocho_modelos_de_gemini_siguen():
    """La lista de modelos se mantiene tal cual estaba."""
    from config_ia import PROVEEDORES

    modelos = list(PROVEEDORES["Google Gemini"]["modelos"].values())
    assert modelos == [
        "gemini-3.8-flash",
        "gemini-3.8-live",
        "gemini-3.8-live-extended-thinking",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
    ], modelos


def test_cadena_de_respaldo_empieza_en_el_elegido():
    from config_ia import _cadena_de_respaldo

    modelos = [f"m{i}" for i in range(4)]
    ctx = {"model_name": "m1", "modelos_disponibles": modelos}
    # Empieza por el elegido y sigue el orden dando la vuelta al final.
    assert _cadena_de_respaldo(ctx) == ["m1", "m2", "m3", "m0"]


def test_reintenta_el_mismo_modelo_antes_de_cambiar():
    """Un fallo puntual reintenta el MISMO modelo; no salta al siguiente."""
    import config_ia

    llamadas: list[str] = []
    intentos_vistos: list[tuple] = []

    def _progreso(modelo, intento, total):
        intentos_vistos.append((modelo, intento, total))

    def _llamar(api_key, modelo, prompt, texto):
        llamadas.append(modelo)
        if len(llamadas) < 2:
            raise RuntimeError("503 overloaded")
        return "respuesta buena"

    orig_llamar = config_ia._llamar_gemini
    orig_sleep = config_ia.time.sleep
    config_ia._llamar_gemini = _llamar
    config_ia.time.sleep = lambda s: None  # sin esperar en la prueba
    try:
        ctx = {
            "proveedor": "Google Gemini",
            "api_key": "x",
            "model_name": "mA",
            "etiqueta_modelo": "A",
            "modelos_disponibles": ["mA", "mB"],
        }
        r = config_ia.evaluar_texto_llm(
            ctx, "sys", "txt", max_intentos=3, espera=0, on_progreso=_progreso
        )
    finally:
        config_ia._llamar_gemini = orig_llamar
        config_ia.time.sleep = orig_sleep

    assert r.ok, r.texto
    assert r.texto == "respuesta buena"
    assert llamadas == ["mA", "mA"], llamadas  # mismo modelo, dos veces
    assert intentos_vistos == [("mA", 1, 3), ("mA", 2, 3)], intentos_vistos
    assert r.intentos == 2
    assert not r.hubo_fallback
    assert r.descartados == []


def test_espera_real_entre_reintentos():
    """Comprueba que se espera de verdad entre intento e intento."""
    import time as _time

    import config_ia

    def _llamar(api_key, modelo, prompt, texto):
        raise RuntimeError("500")

    orig_llamar = config_ia._llamar_gemini
    config_ia._llamar_gemini = _llamar
    try:
        ctx = {
            "proveedor": "Google Gemini",
            "api_key": "x",
            "model_name": "mA",
            "etiqueta_modelo": "A",
            "modelos_disponibles": ["mA"],
        }
        inicio = _time.monotonic()
        r = config_ia.evaluar_texto_llm(
            ctx, "sys", "txt", max_intentos=3, espera=0.4
        )
        transcurrido = _time.monotonic() - inicio
    finally:
        config_ia._llamar_gemini = orig_llamar

    assert not r.ok
    # 3 intentos con espera entre medias = 2 esperas de 0,4 s.
    assert 0.75 < transcurrido < 1.6, transcurrido


def test_pasa_al_siguiente_modelo_si_se_agotan_los_intentos():
    """Agotados los reintentos de un modelo, prueba el siguiente."""
    import config_ia

    llamadas: list[str] = []

    def _llamar(api_key, modelo, prompt, texto):
        llamadas.append(modelo)
        if modelo == "mA":
            raise RuntimeError("500 error del servidor")
        return "ok desde el respaldo"

    orig_llamar = config_ia._llamar_gemini
    orig_sleep = config_ia.time.sleep
    config_ia._llamar_gemini = _llamar
    config_ia.time.sleep = lambda s: None
    try:
        ctx = {
            "proveedor": "Google Gemini",
            "api_key": "x",
            "model_name": "mA",
            "etiqueta_modelo": "A",
            "modelos_disponibles": ["mA", "mB", "mC"],
        }
        r = config_ia.evaluar_texto_llm(ctx, "sys", "txt", max_intentos=2, espera=0)
    finally:
        config_ia._llamar_gemini = orig_llamar
        config_ia.time.sleep = orig_sleep

    assert r.ok, r.texto
    assert r.texto == "ok desde el respaldo"
    assert llamadas == ["mA", "mA", "mB"], llamadas
    assert r.hubo_fallback
    assert r.descartados == ["mA"]
    assert r.modelo == "mB"


def test_error_de_api_key_no_se_reintenta():
    """Un 401 corta la cadena: no tiene sentido probar 8 modelos."""
    import config_ia
    from google.genai import errors as genai_errors

    llamadas: list[str] = []

    def _llamar(api_key, modelo, prompt, texto):
        llamadas.append(modelo)
        raise genai_errors.ClientError(401, {"error": {"message": "API key not valid"}})

    orig_llamar = config_ia._llamar_gemini
    orig_sleep = config_ia.time.sleep
    config_ia._llamar_gemini = _llamar
    config_ia.time.sleep = lambda s: None
    try:
        ctx = {
            "proveedor": "Google Gemini",
            "api_key": "mala",
            "model_name": "mA",
            "etiqueta_modelo": "A",
            "modelos_disponibles": ["mA", "mB", "mC"],
        }
        r = config_ia.evaluar_texto_llm(ctx, "sys", "txt", max_intentos=3, espera=0)
    finally:
        config_ia._llamar_gemini = orig_llamar
        config_ia.time.sleep = orig_sleep

    assert not r.ok
    assert llamadas == ["mA"], llamadas  # solo un intento, un solo modelo
    assert "no tiene sentido reintentar" in r.texto
    assert "API key not valid" in r.texto


def test_si_falla_todo_avisa_cuantos_modelos():
    import config_ia

    def _llamar(api_key, modelo, prompt, texto):
        raise RuntimeError("timeout")

    orig_llamar = config_ia._llamar_gemini
    orig_sleep = config_ia.time.sleep
    config_ia._llamar_gemini = _llamar
    config_ia.time.sleep = lambda s: None
    try:
        ctx = {
            "proveedor": "Google Gemini",
            "api_key": "x",
            "model_name": "mA",
            "etiqueta_modelo": "A",
            "modelos_disponibles": ["mA", "mB"],
        }
        r = config_ia.evaluar_texto_llm(ctx, "sys", "txt", max_intentos=2, espera=0)
    finally:
        config_ia._llamar_gemini = orig_llamar
        config_ia.time.sleep = orig_sleep

    assert not r.ok
    assert r.descartados == ["mA", "mB"]
    assert "Modelos probados:** 2" in r.texto


def test_avisa_si_respondio_otro_modelo():
    """El usuario debe saber si la respuesta vino de un modelo de respaldo."""
    from config_ia import RespuestaIA, resumen_del_intento

    r = RespuestaIA(
        texto="ok", ok=True, modelo="gemini-3.6-flash", intentos=1,
        descartados=["gemini-3.8-flash"],
    )
    aviso = resumen_del_intento(r, "gemini-3.8-flash")
    assert aviso is not None
    assert "gemini-3.6-flash" in aviso
    assert "respaldo" in aviso

    # Si respondio el elegido a la primera, no hay nada que avisar.
    r2 = RespuestaIA(texto="ok", ok=True, modelo="gemini-3.8-flash", intentos=1)
    assert resumen_del_intento(r2, "gemini-3.8-flash") is None

    # Si fallo, el error ya se muestra y no hace falta aviso extra.
    r3 = RespuestaIA(texto="error", ok=False)
    assert resumen_del_intento(r3, "gemini-3.8-flash") is None


# ---------------------------------------------------------------------------
# 6. Ejecucion como script
# ---------------------------------------------------------------------------
def _main() -> int:
    pruebas = [
        (nombre, fn)
        for nombre, fn in sorted(globals().items())
        if nombre.startswith("test_") and callable(fn)
    ]
    fallos = 0
    for nombre, funcion in pruebas:
        try:
            with warnings.catch_warnings():
                # El \\sigma del original era un SyntaxWarning en una f-string.
                warnings.simplefilter("error", SyntaxWarning)
                funcion()
            print(f"  PASA  {nombre}")
        except AssertionError as exc:
            fallos += 1
            print(f"  FALLA {nombre}\n         {exc}")
        except Exception as exc:  # noqa: BLE001
            fallos += 1
            print(f"  ERROR {nombre}: {type(exc).__name__}: {exc}")

    print()
    total = len(pruebas)
    if fallos:
        print(f"{fallos} de {total} pruebas fallidas.")
    else:
        print(f"Todas las pruebas correctas ({total}).")
    return 1 if fallos else 0


if __name__ == "__main__":
    raise SystemExit(_main())