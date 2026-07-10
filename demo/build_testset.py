#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Graba un mini test set leyendo frases en pantalla, de frente y de cerca (caso ideal).
Por cada frase: ESPACIO para grabar, la lees, ESPACIO para cortar -> guarda npz + ref.
Mismo crop que el entrenamiento (MediaPipe -> mean-face -> 96x96 gris 25fps).

APPEND + RESUMIBLE: no pisa clips ya grabados. Lee el manifest existente, saltea las
frases ya hechas y arranca en la primera que falte. Numeracion clip_00..clip_99 (idx de frase).

Corre en el env `ptt`:
  ~/miniconda3/envs/ptt/bin/python build_testset.py [--out ~/vsr_selftest]

Teclas:  ESPACIO=grabar/cortar   r=regrabar la frase actual   n=saltar   q=salir
Al final: los npz + manifest.csv (mergeado) quedan en --out, listos para scorear.
"""
import os, sys, csv, time, argparse
import cv2, numpy as np
# Raiz del repo derivada de este archivo; LABIOS_REPO la pisa (ver .env.example).
REPO = os.environ.get("LABIOS_REPO") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from preprocessing.src.preprocesar import (
    crear_landmarker, detectar_landmarks, cuatro_puntos, remuestrear_a_25fps)
from preprocessing.src.video_process import VideoProcess

# --- frases originales (clip_00..clip_39, YA grabadas) ---
PROMPTS = [
    "hoy me levante temprano y me fui a laburar en bici",
    "che queres que vayamos a tomar unos mates a la plaza",
    "no lo puedo creer se me rompio el celular de nuevo",
    "ayer fui a la cancha y ganamos dos a cero",
    "me parece que va a llover toda la tarde",
    "tengo que estudiar para el parcial de manana",
    "vamos a comer un asado el domingo en casa de mi vieja",
    "el subte estaba lleno y llegue tarde al trabajo",
    "no tengo ganas de cocinar pidamos una pizza",
    "mi hermano se compro un auto usado la semana pasada",
    "hace un calor barbaro prende el ventilador",
    "el finde nos vamos unos dias a la costa con los pibes",
    "me tomo un cafe con leche y tres medialunas",
    "el bondi tardo como media hora en venir",
    "estoy cansado no dormi casi nada anoche",
    "vamos al cine a ver la pelicula nueva",
    "me encanta salir a caminar por el parque",
    "tengo una reunion importante a las tres de la tarde",
    "compre pan y queso para la cena de hoy",
    "mi perro se escapo pero volvio solo a la casa",
    "la semana que viene arrancan las clases otra vez",
    "no encuentro las llaves las busque por todos lados",
    "quedamos en juntarnos el viernes a la noche",
    "el partido se suspendio por la lluvia",
    "me compre unas zapatillas nuevas para correr",
    "estamos organizando una fiesta sorpresa para mi amiga",
    "el tren venia tan lleno que no pude subir",
    "prefiero quedarme en casa y ver una serie",
    "mi vieja hace las mejores empanadas del barrio",
    "ayer charlamos hasta las tres de la manana",
    "me olvide el paraguas y me empape entero",
    "vamos a festejar el cumple en un bar del centro",
    "tengo que ir al banco a pagar unas cuentas",
    "el jefe me pidio que termine el informe hoy",
    "hace mucho que no veo a mis amigos del colegio",
    "me duele la cabeza voy a tomar una aspirina",
    "plantamos unos tomates y albahaca en el balcon",
    "el verano pasado viajamos por toda la patagonia",
    "no me alcanza la plata hasta fin de mes",
    "quiero aprender a tocar la guitarra este ano",
]
# --- frases NUEVAS (clip_40 en adelante). Mismo estilo, sin tildes ni ñ (como las de arriba). ---
PROMPTS += [
    "me quede sin bateria justo cuando te iba a llamar",
    "pasame la receta de las milanesas de tu vieja",
    "el lunes empiezo el gimnasio en serio esta vez",
    "se corto la luz en todo el barrio anoche",
    "vamos a lo de la abuela a almorzar el domingo",
    "me clave una siesta de tres horas y me desperte peor",
    "tenes monedas para el bondi o pago con la sube",
    "la pelicula estaba buenisima pero re larga",
    "me pedi un delivery porque no tenia ganas de nada",
    "juntemos plata entre todos para el regalo",
    "hace un frio tremendo agarra una campera",
    "mi vecino escucha musica fuerte hasta la madrugada",
    "anote todo en un cuaderno para no olvidarme",
    "el finde hay un recital en el parque centenario",
    "se me mancho la remera con salsa de tomate",
    "labure hasta tarde y llegue muerto a casa",
    "carga el termo que salimos a tomar mate",
    "me olvide la clave del banco otra vez",
    "vamos caminando que esta a dos cuadras nomas",
    "compramos fruta y verdura en la feria del sabado",
    "el gato rompio el florero de la mesa",
    "tengo turno con el dentista a las cuatro",
    "prende el horno que hago una pizza casera",
    "se llovio todo el techo de la cocina",
    "te presto el cargador pero despues me lo devolves",
    "armamos un grupo para el trabajo practico",
    "hace meses que quiero pintar la habitacion",
    "el bondi paso de largo y no me vio",
    "me compre un libro nuevo para el viaje",
    "cociname unos fideos con manteca porfa",
    "estaba todo cerrado por el feriado largo",
    "mi hermana se recibio de abogada la semana pasada",
    "bajemos en la proxima estacion que ya llegamos",
    "no me funciona el wifi desde ayer a la tarde",
    "traje facturas para el mate de la tarde",
    "se pincho la rueda de la bici en el camino",
    "vamos a ver a river el fin de semana",
    "me gusta caminar por la costanera al atardecer",
    "tengo que renovar el documento antes de fin de ano",
    "hicimos un asado con todos los primos",
    "el aire acondicionado no anda desde el verano",
    "pasa a buscarme a las ocho por casa",
    "me tome el tren equivocado y termine lejisimo",
    "compre entradas para el teatro el mes que viene",
    "hay que sacar la basura antes de las nueve",
    "me quede dormido en el sillon viendo la tele",
    "juntamos hojas secas del jardin toda la tarde",
    "tenes que probar el helado de dulce de leche",
    "el ascensor esta roto subamos por la escalera",
    "me anote en un curso de fotografia online",
    "llovio tanto que se inundo la esquina",
    "prestame plata hasta el viernes que cobro",
    "fuimos a bailar y volvimos a las seis",
    "se me quemo el arroz mientras hablaba por telefono",
    "armemos una lista para el supermercado",
    "el perro del vecino ladra toda la noche",
    "me duele la espalda de tanto estar sentado",
    "cargue nafta y revise las gomas del auto",
    "vamos a tomar algo despues del laburo",
    "hace mucho que no me tomo unas vacaciones",
]
MIN_FRAMES = 12

def wrap(txt, n=42):
    words, lines, cur = txt.split(), [], ""
    for w in words:
        if len(cur)+len(w)+1 > n: lines.append(cur); cur = w
        else: cur = (cur+" "+w).strip()
    if cur: lines.append(cur)
    return lines or [""]

def crop(frames_bgr, fps, lmk, vproc):
    rgb, pts, det = [], [], 0
    for f in frames_bgr:
        r = np.ascontiguousarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        rgb.append(r); lm = detectar_landmarks(r, lmk)
        if lm is not None: pts.append(cuatro_puntos(lm, r.shape[1], r.shape[0])); det += 1
        else: pts.append(None)
    if det/max(len(rgb),1) < 0.5: return None, det/max(len(rgb),1)
    seq = vproc(rgb, pts)
    if seq is None or len(seq)==0: return None, det/max(len(rgb),1)
    return np.asarray(remuestrear_a_25fps([seq[i] for i in range(len(seq))], fps), dtype=np.uint8), det/max(len(rgb),1)

def cargar_manifest(path):
    """Devuelve dict {clip_name: texto} del manifest existente (o vacio)."""
    rec = {}
    if os.path.exists(path):
        for row in csv.DictReader(open(path, encoding="utf-8")):
            rec[row["clip"]] = row["texto"]
    return rec

def escribir_manifest(path, rec):
    filas = sorted(rec.items(), key=lambda kv: int(kv[0].split("_")[1]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f); wr.writerow(["clip", "texto"]); wr.writerows(filas)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/vsr_selftest"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, "manifest.csv")

    # RESUME: clips ya grabados = estan en el manifest Y el npz existe en disco
    rec = cargar_manifest(manifest_path)
    def ya_grabado(i):
        name = f"clip_{i:02d}"
        return name in rec and os.path.exists(os.path.join(args.out, name + ".npz"))
    total = len(PROMPTS)
    hechos = sum(1 for i in range(total) if ya_grabado(i))
    idx = next((i for i in range(total) if not ya_grabado(i)), total)
    if idx >= total:
        print(f"[testset] ya estan las {total} frases grabadas en {args.out}. Nada que hacer."); return
    print(f"[testset] {hechos}/{total} ya grabadas. Arranco en la #{idx+1} (clip_{idx:02d}).")

    lmk, vproc = crear_landmarker(), VideoProcess()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[testset] ERROR: no abre la camara"); return
    # WARMUP: en macOS el primer read suele fallar mientras la camara arranca
    print("[testset] calentando camara...", flush=True)
    for _ in range(40):
        okw, _f = cap.read()
        if okw: break
        time.sleep(0.2)

    rec_on, frames, t0, tog = False, [], 0.0, 0.0
    status, fails = "Lee la frase de arriba. ESPACIO para grabar.", 0

    while idx < total:
        if ya_grabado(idx):          # saltear las ya hechas (por si quedan huecos)
            idx += 1; continue
        ok, frame = cap.read()
        if not ok:
            fails += 1
            if fails > 80: print("[testset] la camara dejo de responder"); break
            time.sleep(0.05); continue
        fails = 0
        view = frame.copy(); h, w = view.shape[:2]
        cv2.rectangle(view, (0,0), (w,90), (0,0,0), -1)
        cv2.putText(view, f"[{idx+1}/{total}] LEE EN VOZ ALTA:", (15,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1)
        for i, ln in enumerate(wrap(PROMPTS[idx])):
            cv2.putText(view, ln, (15, 55+i*24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        if rec_on:
            frames.append(frame.copy())
            cv2.circle(view, (25, h-30), 11, (0,0,255), -1)
            cv2.putText(view, f"REC {len(frames)}f", (45, h-22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        else:
            cv2.putText(view, status, (15, h-22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 1)
        cv2.imshow("Grabar mini test set", view)

        key = cv2.waitKey(1) & 0xFF; now = time.time()
        if key == ord('q'): break
        if key == ord('n') and not rec_on:
            idx += 1; status = "saltada."; continue
        if key == ord('r') and not rec_on:
            status = "regraba: ESPACIO"; continue
        if key == ord(' ') and (now - tog) > 0.6:
            tog = now
            if not rec_on:
                rec_on, frames, t0 = True, [], now
            else:
                rec_on = False; dt = max(now - t0, 1e-3)
                if len(frames) < MIN_FRAMES:
                    status = f"muy corto ({len(frames)}f), regraba (ESPACIO)"; continue
                arr, ratio = crop(frames, len(frames)/dt, lmk, vproc)
                if arr is None:
                    status = f"cara no detectada (ratio {ratio:.0%}), regraba"; continue
                name = f"clip_{idx:02d}"
                np.savez_compressed(os.path.join(args.out, name+".npz"), rois=arr)
                rec[name] = PROMPTS[idx]
                escribir_manifest(manifest_path, rec)     # persistir despues de CADA clip (resumible)
                print(f"[testset] {name}: {arr.shape[0]}f, cara {ratio:.0%} -> guardado | '{PROMPTS[idx]}'")
                idx += 1; status = "guardada! siguiente frase."
    cap.release(); cv2.destroyAllWindows()
    escribir_manifest(manifest_path, rec)
    print(f"\n[testset] LISTO: {len(rec)} clips totales en {args.out}  (manifest.csv mergeado)")

if __name__ == "__main__":
    main()
