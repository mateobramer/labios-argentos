"""Estado persistente de las sesiones de personalizacion desde la demo web.

Cada persona tiene su propia carpeta.  El manifest registra tanto las tomas que
se aceptaron como los reintentos y frases salteadas: asi la recoleccion se puede
retomar sin perder contexto y los datos quedan auditables para el dataset futuro.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path


NIVELES = (
    (40, "Primer modelo personal", "Meta sugerida: 40 tomas."),
    (60, "Perfil robusto", "Segunda meta: 60 tomas."),
    (61, "Corpus abierto", "Sin limite: cada toma nueva suma a tu perfil."),
)


def nombre_persona(valor: object) -> str:
    """Normaliza un alias para usarlo de forma segura como nombre de carpeta."""
    limpio = re.sub(r"[^a-z0-9_-]", "", str(valor or "").lower().strip())
    return limpio or "persona"


class Sesiones:
    """Pequeno almacenamiento JSON; una unica instancia web evita carreras locales."""

    def __init__(self, raiz: str, frases: list[str]):
        self.raiz = Path(os.path.expanduser(raiz))
        self.frases = frases
        self.lock = threading.Lock()
        self.raiz.mkdir(parents=True, exist_ok=True)

    def carpeta(self, persona: str) -> Path:
        ruta = self.raiz / nombre_persona(persona)
        ruta.mkdir(parents=True, exist_ok=True)
        return ruta

    def _leer(self, persona: str) -> dict:
        ruta = self.carpeta(persona) / "manifest.json"
        if not ruta.exists():
            return {"version": 1, "persona": persona, "creada_en": time.time(),
                    "aceptadas": [], "eventos": [], "asignadas": [], "omitidas": []}
        try:
            return json.loads(ruta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # No arriesgamos los clips si un corte deja un manifest invalido.
            return {"version": 1, "persona": persona, "aceptadas": [], "eventos": [], "asignadas": [], "omitidas": []}

    def _guardar(self, persona: str, datos: dict) -> None:
        destino = self.carpeta(persona) / "manifest.json"
        temporal = destino.with_suffix(".json.tmp")
        temporal.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporal, destino)

    def _usadas_globalmente(self) -> dict[int, int]:
        usadas: dict[int, int] = {}
        for manifest in self.raiz.glob("*/manifest.json"):
            try:
                datos = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for indice in datos.get("asignadas", []):
                usadas[int(indice)] = usadas.get(int(indice), 0) + 1
        return usadas

    def _siguiente(self, datos: dict, desde: int = 0) -> int | None:
        completas = {int(x["frase_id"]) for x in datos.get("aceptadas", [])}
        omitidas = {int(i) for i in datos.get("omitidas", [])}
        pendientes = [i for i in datos.get("asignadas", []) if i not in completas and i not in omitidas]
        if pendientes:
            return int(pendientes[0])
        if not self.frases:
            return None
        usadas = self._usadas_globalmente()
        orden = list(range(max(0, desde), len(self.frases))) + list(range(0, max(0, desde)))
        # Primero frases nunca asignadas; cuando se terminan, las menos repetidas.
        elegido = min(orden, key=lambda i: (usadas.get(i, 0), i))
        if elegido not in datos["asignadas"]:
            datos["asignadas"].append(elegido)
        return elegido

    def estado(self, persona: str, desde: int = 0) -> dict:
        persona = nombre_persona(persona)
        with self.lock:
            datos = self._leer(persona)
            siguiente = self._siguiente(datos, desde)
            self._guardar(persona, datos)
            hechas = len({int(x["frase_id"]) for x in datos["aceptadas"]})
        nivel = next((n for n in reversed(NIVELES) if hechas >= n[0]), NIVELES[0])
        return {"persona": persona, "hechas": hechas, "siguiente": siguiente,
                "nivel": {"meta": nivel[0], "titulo": nivel[1], "detalle": nivel[2]},
                "carpeta": str(self.carpeta(persona))}

    def evento(self, persona: str, tipo: str, frase_id: int, **extra: object) -> None:
        with self.lock:
            datos = self._leer(persona)
            datos["eventos"].append({"tipo": tipo, "frase_id": frase_id, "cuando": time.time(), **extra})
            self._guardar(persona, datos)

    def omitir(self, persona: str, frase_id: int) -> dict:
        """Registra que la frase no se dijo hoy y ofrece otra sin perder el historial."""
        with self.lock:
            datos = self._leer(persona)
            if frase_id not in datos.setdefault("omitidas", []):
                datos["omitidas"].append(frase_id)
            datos["eventos"].append({"tipo": "omitida", "frase_id": frase_id, "cuando": time.time()})
            self._guardar(persona, datos)
        return self.estado(persona, frase_id + 1)

    def aceptar(self, persona: str, pendiente: dict) -> dict:
        """Escribe primero el par de archivos y solo despues actualiza el manifest."""
        frase_id = int(pendiente["frase_id"])
        carpeta = self.carpeta(persona)
        base = carpeta / f"clip_{frase_id:03d}"
        tmp_npz, tmp_txt = str(base) + ".npz.tmp", str(base) + ".txt.tmp"
        import numpy as np
        try:
            with open(tmp_npz, "wb") as archivo:
                np.savez_compressed(archivo, rois=pendiente["rois"])
            Path(tmp_txt).write_text(pendiente["texto"], encoding="utf-8")
            os.replace(tmp_npz, str(base) + ".npz")
            os.replace(tmp_txt, str(base) + ".txt")
        finally:
            for ruta in (tmp_npz, tmp_txt):
                if os.path.exists(ruta):
                    os.unlink(ruta)
        with self.lock:
            datos = self._leer(persona)
            datos["aceptadas"] = [x for x in datos["aceptadas"] if int(x["frase_id"]) != frase_id]
            datos["aceptadas"].append({"frase_id": frase_id, "texto": pendiente["texto"],
                                        "archivo": base.name, "frames": pendiente["frames"],
                                        "duracion_s": pendiente["duracion"], "cuando": time.time()})
            datos["eventos"].append({"tipo": "aceptada", "frase_id": frase_id, "cuando": time.time()})
            self._guardar(persona, datos)
        return self.estado(persona, frase_id + 1)
