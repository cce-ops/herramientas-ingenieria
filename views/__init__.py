"""Vistas de la aplicacion unificada.

Cada modulo expone una funcion `render()` que dibuja el contenido de una
pestana de primer nivel y recibe como argumento el contexto de IA
(devuelto por `config_ia.render_sidebar_ia`).
"""

from __future__ import annotations

from views import diseno_producto, planificador, viabilidad

__all__ = ["diseno_producto", "planificador", "viabilidad"]