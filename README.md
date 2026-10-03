# 🛠️ Herramientas de Ingeniería para Proyectos

Las tres aplicaciones de ingeniería reunidas en **un solo enlace**, sin cambiar
la lógica ni el funcionamiento de ninguna de ellas.

| Herramienta | Origen | Qué hace |
| :--- | :--- | :--- |
| 🛠️ **Diseño de Producto** | `Dise-de-producto` | PDS, AMFE, Ishikawa y QFD (evaluación con IA) |
| 📊 **Planificador** | `planificador-proyectos` | PERT, CPM, Gantt, camino crítico y Zaderenko |
| 💰 **Viabilidad** | `calculadora-viabilidad-proyectos` | VAN, TIR y Payback |

Los tres repositorios originales y sus tres URLs siguen funcionando tal cual:
esta aplicación es aditiva y no modifica nada de lo anterior.

---

## 🚀 Puesta en marcha

```bash
git clone <este-repositorio>
cd <este-repositorio>

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

La aplicación queda en `http://localhost:8501`.

### Pruebas

```bash
python test_app.py             # o: python -m pytest test_app.py -v
```

Las pruebas usan `streamlit.testing.v1.AppTest`, que ejecuta la aplicación
completa sin navegador. Hay 31 pruebas que cubren las tres pestañas, la lógica
de cálculo, la persistencia de resultados entre pestañas y el mecanismo de
reintentos y respaldo de la IA.

---

## ⚙️ La API Key se introduce una sola vez

En la barra lateral, arriba del todo. Se rellena **una vez por sesión** y queda
disponible para toda la aplicación.

Conviene saber que **solo dos de las ocho herramientas necesitan IA** (PDS e
Ishikawa). El planificador y la calculadora son cálculo local puro
(`networkx` y `numpy-financial`), así que nunca piden ninguna clave.

Además, la clave se guarda **por proveedor**: si alternas entre Gemini y Groq y
vuelves a Gemini, tu clave sigue ahí y no hay que reescribirla.

Las respuestas de la IA se guardan en `session_state`, de modo que **no se
vuelven a pedir** al cambiar de pestaña. Esto evita gastar cuota de la API por
accidente.

El único proveedor es **Google Gemini**. La clave se guarda por proveedor, así
que se conserva aunque se recargue la página.

| Proveedor | Dónde se obtiene la clave |
| :--- | :--- |
| Google Gemini | [Google AI Studio](https://aistudio.google.com/) |

---

## 🔁 Reintentos y modelos de respaldo

Si una llamada a Gemini falla, la aplicación **no se rinde**:

1. **Reintenta el mismo modelo** hasta 3 veces, esperando 3 segundos entre
   intento e intento.
2. Si se agotan los intentos, **pasa al siguiente modelo** de la lista y repite
   el proceso. La cadena empieza por el modelo elegido y sigue el orden de la
   lista, dando la vuelta al final para que siempre haya alternativas.
3. Mientras ocurre, el spinner indica en qué modelo y en qué intento va.

Si el fallo no tiene sentido reintentarlo —una API Key incorrecta (401), una
petición inválida (400) o un modelo inexistente (404)— la cadena **corta
inmediatamente**. Reintentar 8 modelos con una clave mal escrita solo haría
esperar al usuario.

Cuando la respuesta viene de un modelo distinto del elegido, o tras varios
intentos, la aplicación lo avisa explícitamente: con este mecanismo se puede
estar leyendo una respuesta de un modelo que no es el que se quería, y conviene
saberlo.

Los valores se ajustan arriba de `config_ia.py`:

```python
ESPERA_REINTENTO_SEG = 3.0        # espera entre reintentos
MAX_INTENTOS_POR_MODELO = 3       # intentos por modelo
CODIGOS_SIN_REINTENTO = {400, 401, 403, 404}
```

---

## 📂 Estructura

```text
.
├── app.py                  # punto de entrada: config, barra lateral y pestañas
├── config_ia.py            # proveedores de IA + API Key + única llamada de red
├── views/
│   ├── __init__.py
│   ├── comun.py            # persistencia de resultados y figuras
│   ├── diseno_producto.py  # PDS · AMFE · Ishikawa · QFD
│   ├── planificador.py     # Modo básico (AON) · Modo avanzado (Zaderenko)
│   └── viabilidad.py       # VAN · TIR · Payback
├── requirements.txt
└── test_app.py
```

`app.py` hace tres cosas y ninguna más: configura la página **una vez** (antes
cada app tenía la suya), dibuja la configuración de IA **una vez**, y reparte
las tres herramientas en tres pestañas. Cada herramienta conserva su código.

---

## ⚠️ Un detalle que conviene conocer

Al unir las tres aplicaciones en una sola página, cualquier interacción en una
pestaña relanza **toda** la aplicación (así funciona Streamlit). Las apps
originales mostraban sus resultados con este patrón:

```python
if st.button("Calcular"):
    st.success(resultado)      # <- solo se ve en la ejecución del clic
```

Un botón solo devuelve `True` en la ejecución inmediatamente posterior al clic.
Por eso, en una página única, el Gantt o el AMFE **desaparecerían** en cuanto
el usuario tocara algo en otra pestaña. Se comprobó que ese era el caso:

```text
pulsado "Calcular"          -> resultado visible
editado en otra pestaña     -> resultado DESAPARECIDO
vuelve a su pestaña         -> resultado SIGUE DESAPARECIDO
```

La solución es `views/comun.py`: el resultado se guarda en `session_state` y se
vuelve a pintar desde ahí, en lugar de depender del `return` del botón.

```python
if boton_calcular("Calcular Proyecto Básico", clave="plan__basico", resultado=R_BASICO):
    guardar_resultado(R_BASICO, _calcular_basico(df, fecha))

resultado = leer_resultado(R_BASICO)
if resultado is not None:
    _pintar_basico(resultado, fecha)
```

Es un detalle interno: la interfaz final es la misma, con la ventaja de que los
resultados ya no se pierden. Cada cálculo lleva además un botón **🗑️ Limpiar**.

---

## 🔧 Cambios respecto a las apps originales

Además de la unificación:

1. **Pestañas perezosas.** `st.tabs(..., on_change="rerun")`, disponible desde
   Streamlit 1.55.0, hace que solo se ejecute la pestaña visible. Con las tres
   herramientas en la misma página esto evita recalcular todo en cada tecla.
2. **`google-generativeai` → `google-genai`.** El SDK antiguo ya no recibe
   actualizaciones y lo anuncia así en su propio `import`. Al quedar solo
   Gemini, la dependencia `openai` (usada para NIM, OpenRouter y Groq) ya no es
   necesaria y se ha retirado de `requirements.txt`.
3. **`use_container_width` → `width="stretch"`.** El parámetro anterior estaba
   obsoleto y su fecha de eliminación (2025-12-31) ya había pasado.
4. **Matriz de Zaderenko serializable.** Mezclaba números con `""` y usaba
   etiquetas de nodo numéricas junto a `t_i*` / `t_j`, así que `pyarrow` no
   conseguía convertirla y Streamlit tenía que parchearla. Ahora toda la tabla
   es de texto. El aspecto no cambia.
5. **`\sigma` corregido.** El original escribía `f"Varianza ($\sigma^2$)"`; en
   una f-string, `\s` no es un escape válido y Python emitía un
   `SyntaxWarning`.
6. **Figuras cerradas con `plt.close()`.** Sin esto Streamlit avisa a partir de
   unas 20 figuras abiertas en la misma sesión, fácil de alcanzar al combinar
   las tres apps.
7. **`npf.irr()` se llama una vez** en lugar de dos en la misma expresión.

### Un cambio de disposición

Los tres parámetros generales de la calculadora (**Inversión Inicial**, **Tasa
de Descuento** y **Vida Útil**) han pasado de la barra lateral al interior de su
pestaña. Es el único cambio de posición en la interfaz, y es necesario para que
la barra lateral quede dedicada a la configuración de IA, que es lo que se
rellena una sola vez.

### Una discrepancia que **no** se ha tocado

En AMFE, el texto de ayuda dice:

> Si la **Severidad es ≥ 9** o el **NPR ≥ 100**, marca 🔴 Acción Urgente.

Pero el código marca 🔴 **solo** cuando `S ≥ 9`. Un `NPR ≥ 100` con `S < 9` cae
en 🟠 Prioridad Media-Alta, porque las condiciones de `np.select` se evalúan en
orden y la segunda nunca llega a aplicarse. Se ha mantenido tal cual para no alterar
el funcionamiento; si prefieres que el código haga caso al texto, es un cambio
de una línea.

---

## 🧰 Tecnologías

Python · Streamlit · Pandas · NumPy · NumPy Financial · SciPy · NetworkX ·
Plotly · Matplotlib · Google Gen AI SDK

---

## 📄 Licencia

El repositorio no especifica licencia. Si va a distribuirse o reutilizarse
públicamente, conviene añadir un archivo `LICENSE`.