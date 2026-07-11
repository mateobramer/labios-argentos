#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fase LLM + scoring de la grilla CER/LLM completa.
Para cada celda (JSON de candidatas top-5): corre qwen en 2 tecnicas persistiendo en el
mismo JSON — `corr` (correccion 1-best, prompt Fase 0) y `resc` (n-best rescoring) — y
computa 1-best / corr / resc / oracle-5 con WER/CER + bootstrap pareado de ambos deltas.

Corre con python3 de sistema (solo stdlib). Reanudable: saltea lo que ya tiene corr/resc.
  python3 grid_score.py            # procesa todas las celdas disponibles
  python3 grid_score.py --solo-tabla   # no llama a qwen, solo scorea lo que hay
"""
import os, re, json, glob, random, argparse, unicodedata, urllib.request

AQUI = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(AQUI, "data")
QMODEL = "qwen3:4b-instruct-2507-q4_K_M"
OLLAMA = "http://127.0.0.1:11434"

SYS_RESC = ("Te doy VARIAS transcripciones candidatas de la MISMA frase (lectura de labios, rioplatense). "
            "Se complementan: la palabra correcta suele estar en alguna. Combinalas para la frase mas probable, "
            "sin inventar. Responde SOLO la frase, una linea.")
SYS_CORR = ("Sos un corrector de transcripciones de lectura de labios (lip reading) en espanol "
            "rioplatense (Argentina). Recibis una transcripcion CRUDA y con errores producida por "
            "un modelo de lip reading (sin acentos ni puntuacion). Devolve la version mas probable "
            "de lo que la persona quiso decir, corrigiendo ortografia, concordancia y formas "
            "rioplatenses (voseo: tenes, queres, vos, che).\n"
            "REGLAS ESTRICTAS:\n"
            "1. NO agregues informacion ni palabras que no esten sugeridas por el texto.\n"
            "2. Mante el mismo contenido y un largo parecido; no inventes frases nuevas.\n"
            "3. Si una parte es incomprensible, dejala lo mas parecida posible al original en vez de inventar.\n"
            "4. Responde SOLO con el texto corregido, en una sola linea, sin comillas ni explicaciones.")

def norm(s):
    s = s.lower().strip().replace("ñ", "\x00")
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.replace("\x00", "ñ")
    s = re.sub(r"[^a-z0-9ñ ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def ed(a, b):
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    p = list(range(m + 1))
    for i in range(1, n + 1):
        c = [i] + [0] * m
        for j in range(1, m + 1):
            c[j] = min(p[j] + 1, c[j - 1] + 1, p[j - 1] + (0 if a[i - 1] == b[j - 1] else 1))
        p = c
    return p[m]

def rate(pairs, lvl="w"):
    te = tn = 0
    for r, h in pairs:
        a, b = (r.split(), h.split()) if lvl == "w" else (r, h)
        te += ed(a, b); tn += len(a)
    return 100.0 * te / max(tn, 1)

def qwen(system, user):
    body = {"model": QMODEL, "stream": False, "think": False, "keep_alive": "10m",
            "options": {"temperature": 0.0, "num_predict": 256},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    req = urllib.request.Request(OLLAMA + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        o = json.loads(r.read().decode())
    txt = re.sub(r"<think>.*?</think>", " ", o.get("message", {}).get("content", ""), flags=re.DOTALL)
    return norm(txt)

def textos(d):
    return [c[0] if isinstance(c, list) else c for c in d["cands"] if (c[0] if isinstance(c, list) else c)][:5]

def celdas():
    """Devuelve {nombre_celda: (path, subkey|None)} de todos los JSON disponibles."""
    out = {}
    # fede fp32 (con resc ya hecho) + fede int8 (dict con dos brazos)
    if os.path.exists(DATA + "/cands100_scored.json"):
        out["visper-fp32/fede"] = (DATA + "/cands100_scored.json", None)
    for p in sorted(glob.glob(DATA + "/cands_*.json")):
        b = os.path.basename(p)[len("cands_"):-len(".json")]
        if b == "amigo":
            out["visper-fp32/amigo"] = (p, None)
        elif b.startswith("658_") or b.endswith(("_fp32", "_int8", "_ft05", "_ft07")) or "_" in b:
            test, mod = b.split("_", 1)
            tag = {"fp32": "visper-fp32", "int8": "visper-int8", "ft05": "ft05", "ft07": "ft07"}.get(mod, mod)
            out[f"{tag}/{test}"] = (p, None)
    return out

def cargar(path, sub):
    data = json.load(open(path, encoding="utf-8"))
    if sub is not None:
        data = data[sub]
        for d in data:
            d.setdefault("ref", d.get("ref", ""))
    return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-tabla", action="store_true")
    ap.add_argument("--solo", nargs="*", default=None,
                    help="basenames de JSON a procesar (evita tocar archivos que otra cola sigue escribiendo)")
    args = ap.parse_args()
    global celdas
    if args.solo is not None:
        base = celdas()
        permitidos = set(args.solo)
        filtrado = {k: v for k, v in base.items() if os.path.basename(v[0]) in permitidos}
        celdas = lambda: filtrado
    puntos = []
    print(f"{'celda':22} {'n':>4} {'CER0':>6} {'WER0':>7} {'corr':>7} {'resc':>7} {'orac':>7}  delta_resc (IC95)        delta_corr")
    for nombre, (path, sub) in sorted(celdas().items()):
        data = cargar(path, sub)
        data = [d for d in data if textos(d)]
        if not args.solo_tabla:
            pend = [d for d in data if "resc" not in d or "corr" not in d]
            if pend:
                print(f"[{nombre}] qwen sobre {len(pend)} clips...", flush=True)
            for k, d in enumerate(data):
                cambio = False
                if "corr" not in d:
                    try: d["corr"] = qwen(SYS_CORR, textos(d)[0]); cambio = True
                    except Exception as e: print(f"  corr err {k}: {e}"); continue
                if "resc" not in d:
                    try: d["resc"] = qwen(SYS_RESC, "Candidatas:\n" + "\n".join(
                        f"{i+1}. {c}" for i, c in enumerate(textos(d))) + "\n\nFrase:"); cambio = True
                    except Exception as e: print(f"  resc err {k}: {e}"); continue
                if cambio and k % 10 == 0:
                    full = json.load(open(path, encoding="utf-8"))
                    if sub is not None: full[sub] = data
                    else: full = data
                    json.dump(full, open(path, "w"), ensure_ascii=False)
            full = json.load(open(path, encoding="utf-8"))
            if sub is not None: full[sub] = data
            else: full = data
            json.dump(full, open(path, "w"), ensure_ascii=False)
        listos = [d for d in data if "resc" in d and "corr" in d]
        if not listos:
            print(f"{nombre:22} (sin qwen todavia, {len(data)} cands)"); continue
        p1 = [(d["ref"], textos(d)[0]) for d in listos]
        pc = [(d["ref"], d["corr"]) for d in listos]
        pr = [(d["ref"], d["resc"]) for d in listos]
        po = [(d["ref"], min(textos(d), key=lambda c: ed(d["ref"].split(), c.split()))) for d in listos]
        random.seed(1234)
        n = len(p1); dr = []; dc = []
        for _ in range(5000):
            idx = [random.randrange(n) for _ in range(n)]
            w1 = rate([p1[i] for i in idx])
            dr.append(w1 - rate([pr[i] for i in idx]))
            dc.append(w1 - rate([pc[i] for i in idx]))
        dr.sort(); dc.sort()
        lo, hi = dr[124], dr[4874]; loc, hic = dc[124], dc[4874]
        cer0, wer0 = rate(p1, "c"), rate(p1)
        drm, dcm = wer0 - rate(pr), wer0 - rate(pc)
        sig = "✅" if lo > 0 else ("❌sig" if hi < 0 else "≈")
        sigc = "✅" if loc > 0 else ("❌sig" if hic < 0 else "≈")
        print(f"{nombre:22} {n:>4} {cer0:6.2f} {wer0:7.2f} {rate(pc):7.2f} {rate(pr):7.2f} {rate(po):7.2f}  "
              f"{drm:+.2f} [{lo:+.2f},{hi:+.2f}]{sig}   {dcm:+.2f}{sigc}")
        puntos.append({"celda": nombre, "n": n, "cer_base": round(cer0, 2), "wer_base": round(wer0, 2),
                       "wer_corr": round(rate(pc), 2), "wer_resc": round(rate(pr), 2),
                       "wer_oracle": round(rate(po), 2), "delta_resc": round(drm, 2),
                       "ic_resc": [round(lo, 2), round(hi, 2)], "delta_corr": round(dcm, 2),
                       "ic_corr": [round(loc, 2), round(hic, 2)]})
    json.dump(puntos, open(DATA + "/grid_puntos.json", "w"), ensure_ascii=False, indent=1)
    # ejemplos por celda (para el paper): 4 con mayor mejora del rescoring + 2 al azar
    ejemplos = {}
    for nombre, (path, sub) in sorted(celdas().items()):
        data = [d for d in cargar(path, sub) if "resc" in d and "corr" in d and textos(d)]
        if not data:
            continue
        def mejora(d):
            r = d["ref"].split()
            return ed(r, textos(d)[0].split()) - ed(r, d["resc"].split())
        top = sorted(data, key=mejora, reverse=True)[:4]
        random.seed(7)
        extra = random.sample(data, min(2, len(data)))
        ejemplos[nombre] = [{"ref": d["ref"], "cands": textos(d), "corr_1best": d["corr"],
                             "resc_nbest": d["resc"]} for d in top + extra]
    json.dump({"prompts": {"rescoring_nbest": SYS_RESC, "correccion_1best": SYS_CORR,
                           "modelo": QMODEL, "temperatura": 0.0, "think": False},
               "ejemplos": ejemplos},
              open(DATA + "/grid_prompts_y_ejemplos.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n[grid] {len(puntos)} celdas -> grid_puntos.json + grid_prompts_y_ejemplos.json")

if __name__ == "__main__":
    main()
