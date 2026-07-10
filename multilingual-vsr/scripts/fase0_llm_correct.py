#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""FASE 0 — Correccion offline con LLM local (qwen3:4b via Ollama) sobre las hipotesis
de VSR ya generadas (test.inf con lineas 'ref#hyp', ya normalizadas). Mide WER/CER
crudo vs corregido y estratifica por CER del clip para testear la hipotesis:
  'un CER bajo se corrige mejor que un WER bajo; con CER alto el LLM alucina y empeora'.

Metricas IDENTICAS a zeroshot.py (norm/edits/corpus_rate/bootstrap_ci) para comparabilidad.

Uso:
  python fase0_llm_correct.py --inf runs/ft05b_test.inf --tag ft05b [--model qwen3:4b]
      [--limit N] [--host http://127.0.0.1:11434] [--out-prefix runs/fase0]
"""
import os, re, csv, json, time, argparse, unicodedata, random, urllib.request

# ------------- metricas (copiadas 1:1 de zeroshot.py) -------------
def norm(s):
    s = s.lower().strip()
    s = s.replace("ñ", "\x00")
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.replace("\x00", "ñ")
    s = re.sub(r"[^a-z0-9ñ ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def edits(a, b):
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]

def corpus_rate(pairs, level):
    tot_e = tot_n = 0; per = []
    for ref, hyp in pairs:
        r, h = (ref.split(), hyp.split()) if level == "word" else (ref, hyp)
        e = edits(r, h); n = len(r)
        tot_e += e; tot_n += n; per.append((e, n))
    return 100.0 * tot_e / max(tot_n, 1), per

def bootstrap_ci(per, iters=2000, seed=1234):
    rnd = random.Random(seed); N = len(per); vals = []
    for _ in range(iters):
        te = tn = 0
        for _ in range(N):
            e, n = per[rnd.randrange(N)]; te += e; tn += n
        vals.append(100.0 * te / max(tn, 1))
    vals.sort()
    return (vals[int(0.975*iters)] - vals[int(0.025*iters)]) / 2.0

# ------------- corrector LLM (Ollama /api/chat, think=false) -------------
SYSTEM = (
    "Sos un corrector de transcripciones de lectura de labios (lip reading) en espanol "
    "rioplatense (Argentina). Recibis una transcripcion CRUDA y con errores producida por "
    "un modelo de lip reading (sin acentos ni puntuacion). Devolve la version mas probable "
    "de lo que la persona quiso decir, corrigiendo ortografia, concordancia y formas "
    "rioplatenses (voseo: tenes, queres, vos, che).\n"
    "REGLAS ESTRICTAS:\n"
    "1. NO agregues informacion ni palabras que no esten sugeridas por el texto.\n"
    "2. Mante el mismo contenido y un largo parecido; no inventes frases nuevas.\n"
    "3. Si una parte es incomprensible, dejala lo mas parecida posible al original en vez de inventar.\n"
    "4. Responde SOLO con el texto corregido, en una sola linea, sin comillas ni explicaciones."
)

def correct(host, model, hyp, timeout=120):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": hyp},
        ],
        "stream": False,
        "think": False,
        "keep_alive": "10m",
        "options": {"temperature": 0.0, "num_predict": 256, "top_p": 0.9},
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(host + "/api/chat", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read().decode("utf-8"))
    txt = out.get("message", {}).get("content", "")
    txt = re.sub(r"<think>.*?</think>", " ", txt, flags=re.DOTALL)  # por si el think se filtra
    return txt.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inf", required=True, help="archivo con lineas ref#hyp (ya normalizadas)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="qwen3:4b-instruct-2507-q4_K_M")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-prefix", default="runs/fase0")
    args = ap.parse_args()

    lines = [ln.rstrip("\n") for ln in open(args.inf, encoding="utf-8") if "#" in ln]
    if args.limit: lines = lines[:args.limit]
    refs, hyps = [], []
    for ln in lines:
        r, h = ln.split("#", 1)
        refs.append(norm(r)); hyps.append(norm(h))
    print(f"[fase0] {len(lines)} clips | modelo={args.model} | tag={args.tag}", flush=True)

    corr = []; rows_out = []
    t0 = time.time()
    for i, h in enumerate(hyps, 1):
        try:
            c = norm(correct(args.host, args.model, h))
        except Exception as e:
            c = h; print(f"  [warn] clip {i} fallo LLM: {e}", flush=True)
        corr.append(c)
        rows_out.append(f"{refs[i-1]}#{h}#{c}")
        if i % 10 == 0 or i == len(hyps):
            el = time.time() - t0
            wb, _ = corpus_rate(list(zip(refs[:i], hyps[:i])), "word")
            wc, _ = corpus_rate(list(zip(refs[:i], corr)), "word")
            print(f"  {i}/{len(hyps)} WER base={wb:.2f} corr={wc:.2f} "
                  f"({el/i:.2f}s/clip, ETA {el/i*(len(hyps)-i)/60:.1f}min)", flush=True)

    base_pairs = list(zip(refs, hyps))
    corr_pairs = list(zip(refs, corr))
    wb, pwb = corpus_rate(base_pairs, "word"); cb, pcb = corpus_rate(base_pairs, "char")
    wc, pwc = corpus_rate(corr_pairs, "word"); cc, pcc = corpus_rate(corr_pairs, "char")

    # ---- estratificacion por CER-por-clip del baseline (testea la hipotesis) ----
    def clip_cer(ref, hyp):
        return 100.0 * edits(ref, hyp) / max(len(ref), 1)
    buckets = [(0,20),(20,40),(40,60),(60,200)]
    strata = []
    for lo, hi in buckets:
        idx = [k for k in range(len(refs)) if lo <= clip_cer(refs[k], hyps[k]) < hi]
        if not idx:
            strata.append((lo, hi, 0, None, None, None)); continue
        bp = [base_pairs[k] for k in idx]; cp = [corr_pairs[k] for k in idx]
        wbb, _ = corpus_rate(bp, "word"); wcc, _ = corpus_rate(cp, "word")
        strata.append((lo, hi, len(idx), wbb, wcc, wcc - wbb))

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    pref = f"{args.out_prefix}_{args.tag}"
    with open(pref + "_corr.inf", "w", encoding="utf-8") as f:
        f.write("\n".join(rows_out) + "\n")   # ref#hyp_base#hyp_corr

    L = []
    L.append(f"# FASE 0 — correccion LLM ({args.model}) sobre {args.tag}  (N={len(refs)})")
    L.append(f"%WER base:      {wb:.4f} ± {bootstrap_ci(pwb):.4f}")
    L.append(f"%WER corregido: {wc:.4f} ± {bootstrap_ci(pwc):.4f}   (delta {wc-wb:+.4f})")
    L.append(f"%CER base:      {cb:.4f} ± {bootstrap_ci(pcb):.4f}")
    L.append(f"%CER corregido: {cc:.4f} ± {bootstrap_ci(pcc):.4f}   (delta {cc-cb:+.4f})")
    L.append("")
    L.append("# Estratos por CER-por-clip del baseline (hipotesis: CER bajo mejora, CER alto empeora)")
    L.append(f"{'rango_CER':>10} {'n':>5} {'WER_base':>9} {'WER_corr':>9} {'delta':>8}")
    for lo, hi, n, wbb, wcc, d in strata:
        if n == 0:
            L.append(f"{f'{lo}-{hi}':>10} {n:>5} {'-':>9} {'-':>9} {'-':>8}"); continue
        L.append(f"{f'{lo}-{hi}':>10} {n:>5} {wbb:>9.2f} {wcc:>9.2f} {d:>+8.2f}")
    txt = "\n".join(L) + "\n"
    with open(pref + ".wer", "w", encoding="utf-8") as f:
        f.write(txt)
    print("\n" + txt)
    print("ejemplos (ref | base | corr):")
    for r in rows_out[:8]:
        a, b, c = r.split("#", 2)
        print(f"  REF : {a}\n  BASE: {b}\n  CORR: {c}\n")

if __name__ == "__main__":
    main()
