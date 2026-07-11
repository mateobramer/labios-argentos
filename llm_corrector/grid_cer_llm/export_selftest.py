import csv, os, numpy as np
SP = os.path.dirname(os.path.abspath(__file__))
base = os.path.join(SP, 'gimeno_selftest', 'Selftest')
rois_out, text_out = base + '/ROIs', base + '/transcriptions'
splits_out = base + '/splits/zero-shot'
for d in (rois_out, text_out, splits_out): os.makedirs(d, exist_ok=True)
fuentes = [('s01', os.path.expanduser('~/vsr_selftest')), ('s02', os.path.expanduser('~/Desktop/grabaciones'))]
ids, mapeo = [], []
for spk, d in fuentes:
    os.makedirs(f'{rois_out}/{spk}', exist_ok=True); os.makedirs(f'{text_out}/{spk}', exist_ok=True)
    rows = list(csv.DictReader(open(f'{d}/manifest.csv', encoding='utf-8')))
    for i, r in enumerate(rows):
        rois = np.load(f"{d}/{r['clip']}.npz")['rois']
        sid = f'{spk}_{i:04d}'
        np.savez_compressed(f'{rois_out}/{spk}/{sid}.npz', rois=rois)
        open(f'{text_out}/{spk}/{sid}.txt', 'w', encoding='utf-8').write(r['texto'] + '\n')
        ids.append(sid); mapeo.append([sid, spk, r['clip'], len(rois), r['texto']])
with open(f'{splits_out}/testSelftest.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(['sampleID']); [w.writerow([s]) for s in ids]
with open(f'{base}/mapeo.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(['sampleID','spk','clip','n_frames','texto']); w.writerows(mapeo)
print(f'exportados {len(ids)} clips (s01={sum(1 for s in ids if s.startswith("s01"))}, s02={sum(1 for s in ids if s.startswith("s02"))})')
