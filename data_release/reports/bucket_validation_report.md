# Bucket validation report

bucket: gs://labios-argentos-vsr-clean-v1/
iam_fg_object_viewer: true
resource_filter: (name~vsr-full-clean OR name~vsr-cleaning-vm OR labels.task=full-clean-release)
remaining_vms_matching_filter: "vsr-full-clean-gpu-normal-20260707-2205"
remaining_disks_matching_filter: "vsr-full-clean-gpu-normal-20260707-2205"
remaining_static_addresses_matching_filter: ""

## Counts

- argentina_existing_mp4: 12112
- argentina_existing_npz: 12112
- argentina_existing_large_txt: 14305
- argentina_existing_clean_gpt_v1_txt: 0
- argentina_existing_turbo_txt: 2193
- argentina_existing_large_reconstructed_txt: 14305
- argentina_existing_clips_with_audio: 2193
- argentina_existing_reconstructed_audio: 2193
- spanish_general_mp4: 10356
- spanish_general_npz: 42599
- spanish_general_large_txt: 46991
- spanish_general_turbo_txt: 0
- context_packs: 61
- argentina_new_discovery_source_videos: 20
- argentina_new_discovery_source_audio: 20
- argentina_new_discovery_metadata: 41
- argentina_new_discovery_clips_with_audio: 13193
- argentina_new_discovery_rois_npz: 2248
- argentina_new_discovery_roi_mp4: 2248
- argentina_new_discovery_large_txt: 13193
- argentina_new_discovery_turbo_txt: 13193

## Manifest rows

- argentina_existing_manifest_rows: 9191
- argentina_new_manifest_rows: 20
- spanish_general_manifest_rows: 47152
- clean_manifest_rows: 9191
- alignment_manifest_rows: 9191
- existing_reconstruction_manifest_rows: 2193
- asr_large_turbo_manifest_rows: 4392
- final_release_manifest_rows: 22384
- final_train_manifest_rows: 10315
- final_eval_manifest_rows: 1124
- new_discovery_ingest_manifest_rows: 20
- new_discovery_clip_manifest_rows: 13706
- new_discovery_asr_manifest_rows: 26386
- new_discovery_roi_manifest_rows: 13193
- spanish_general_asr_manifest_rows: 47152

## MP4 samples

- gs://labios-argentos-vsr-clean-v1/argentina/existing/clips_mp4/24HS EN LOS HOTELES MÁS RAROS DE ARGENTINA/clip_0014.mp4 video=true audio=false audio_zero_marker=false
- gs://labios-argentos-vsr-clean-v1/argentina/existing/clips_mp4/24HS EN LOS HOTELES MÁS RAROS DE ARGENTINA/clip_0029.mp4 video=true audio=false audio_zero_marker=false
- gs://labios-argentos-vsr-clean-v1/argentina/existing/clips_mp4/24HS EN LOS HOTELES MÁS RAROS DE ARGENTINA/clip_0034.mp4 video=true audio=false audio_zero_marker=false
- gs://labios-argentos-vsr-clean-v1/argentina/existing/clips_mp4/24HS EN LOS HOTELES MÁS RAROS DE ARGENTINA/clip_0035.mp4 video=true audio=false audio_zero_marker=false
- gs://labios-argentos-vsr-clean-v1/argentina/existing/clips_mp4/24HS EN LOS HOTELES MÁS RAROS DE ARGENTINA/clip_0038.mp4 video=true audio=false audio_zero_marker=false

## Reconstructed MP4 samples

- gs://labios-argentos-vsr-clean-v1/argentina/existing/clips_with_audio/f01__AN_CDOTA_VIAJE_MUNDIAL_BRASIL_2014_parte_1/clip_0001.mp4 video=true audio=true audio_zero_marker=false
- gs://labios-argentos-vsr-clean-v1/argentina/existing/clips_with_audio/f01__AN_CDOTA_VIAJE_MUNDIAL_BRASIL_2014_parte_1/clip_0002.mp4 video=true audio=true audio_zero_marker=false
- gs://labios-argentos-vsr-clean-v1/argentina/existing/clips_with_audio/f01__AN_CDOTA_VIAJE_MUNDIAL_BRASIL_2014_parte_1/clip_0003.mp4 video=true audio=true audio_zero_marker=false
- gs://labios-argentos-vsr-clean-v1/argentina/existing/clips_with_audio/f01__AN_CDOTA_VIAJE_MUNDIAL_BRASIL_2014_parte_1/clip_0006.mp4 video=true audio=true audio_zero_marker=false
- gs://labios-argentos-vsr-clean-v1/argentina/existing/clips_with_audio/f01__AN_CDOTA_VIAJE_MUNDIAL_BRASIL_2014_parte_1/clip_0007.mp4 video=true audio=true audio_zero_marker=false

## New discovery source samples

- source_video gs://labios-argentos-vsr-clean-v1/argentina/new_discovery/source_videos/EhYznjqlcKY/EhYznjqlcKY.f135.mp4 video=true audio=false audio_zero_marker=false
- source_video gs://labios-argentos-vsr-clean-v1/argentina/new_discovery/source_videos/EuSM3LscaWI/EuSM3LscaWI.f398.mp4 video=true audio=false audio_zero_marker=false
- source_video gs://labios-argentos-vsr-clean-v1/argentina/new_discovery/source_videos/IGYG0Kn0wxo/IGYG0Kn0wxo.f243.webm video=true audio=false audio_zero_marker=false
- source_video gs://labios-argentos-vsr-clean-v1/argentina/new_discovery/source_videos/ITovsJg-q5c/ITovsJg-q5c.f398.mp4 video=true audio=false audio_zero_marker=false
- source_video gs://labios-argentos-vsr-clean-v1/argentina/new_discovery/source_videos/JixCyhEGE0A/JixCyhEGE0A.f136.mp4 video=true audio=false audio_zero_marker=false
- source_audio gs://labios-argentos-vsr-clean-v1/argentina/new_discovery/source_audio/EhYznjqlcKY/EhYznjqlcKY.f251.webm video=false audio=true audio_zero_marker=false
- source_audio gs://labios-argentos-vsr-clean-v1/argentina/new_discovery/source_audio/EuSM3LscaWI/EuSM3LscaWI.f251.webm video=false audio=true audio_zero_marker=false
- source_audio gs://labios-argentos-vsr-clean-v1/argentina/new_discovery/source_audio/IGYG0Kn0wxo/IGYG0Kn0wxo.f243.webm video=true audio=false audio_zero_marker=false
- source_audio gs://labios-argentos-vsr-clean-v1/argentina/new_discovery/source_audio/ITovsJg-q5c/ITovsJg-q5c.f251.webm video=false audio=true audio_zero_marker=false
- source_audio gs://labios-argentos-vsr-clean-v1/argentina/new_discovery/source_audio/JixCyhEGE0A/JixCyhEGE0A.f251.webm video=false audio=true audio_zero_marker=false

## New discovery clips_with_audio samples

- gs://labios-argentos-vsr-clean-v1/argentina/new_discovery/clips_with_audio/EhYznjqlcKY/clip_0000.mp4 video=true audio=true audio_zero_marker=false
- gs://labios-argentos-vsr-clean-v1/argentina/new_discovery/clips_with_audio/EhYznjqlcKY/clip_0001.mp4 video=true audio=true audio_zero_marker=false
- gs://labios-argentos-vsr-clean-v1/argentina/new_discovery/clips_with_audio/EhYznjqlcKY/clip_0002.mp4 video=true audio=true audio_zero_marker=false
- gs://labios-argentos-vsr-clean-v1/argentina/new_discovery/clips_with_audio/EhYznjqlcKY/clip_0003.mp4 video=true audio=true audio_zero_marker=false
- gs://labios-argentos-vsr-clean-v1/argentina/new_discovery/clips_with_audio/EhYznjqlcKY/clip_0004.mp4 video=true audio=true audio_zero_marker=false

## NPZ samples

- gs://labios-argentos-vsr-clean-v1/argentina/existing/rois_npz/24HS EN LOS HOTELES MÁS RAROS DE ARGENTINA/clip_0014.npz key=rois shape=255x96x96 dtype=uint8
- gs://labios-argentos-vsr-clean-v1/argentina/existing/rois_npz/24HS EN LOS HOTELES MÁS RAROS DE ARGENTINA/clip_0029.npz key=rois shape=85x96x96 dtype=uint8
- gs://labios-argentos-vsr-clean-v1/argentina/existing/rois_npz/24HS EN LOS HOTELES MÁS RAROS DE ARGENTINA/clip_0034.npz key=rois shape=201x96x96 dtype=uint8
- gs://labios-argentos-vsr-clean-v1/argentina/existing/rois_npz/24HS EN LOS HOTELES MÁS RAROS DE ARGENTINA/clip_0035.npz key=rois shape=107x96x96 dtype=uint8
- gs://labios-argentos-vsr-clean-v1/argentina/existing/rois_npz/24HS EN LOS HOTELES MÁS RAROS DE ARGENTINA/clip_0038.npz key=rois shape=185x96x96 dtype=uint8

## New discovery NPZ samples

- gs://labios-argentos-vsr-clean-v1/argentina/new_discovery/rois_npz/EhYznjqlcKY/clip_0000.npz key=rois shape=151x96x96 dtype=uint8
- gs://labios-argentos-vsr-clean-v1/argentina/new_discovery/rois_npz/EhYznjqlcKY/clip_0002.npz key=rois shape=151x96x96 dtype=uint8
- gs://labios-argentos-vsr-clean-v1/argentina/new_discovery/rois_npz/EhYznjqlcKY/clip_0005.npz key=rois shape=151x96x96 dtype=uint8
- gs://labios-argentos-vsr-clean-v1/argentina/new_discovery/rois_npz/EhYznjqlcKY/clip_0006.npz key=rois shape=151x96x96 dtype=uint8
- gs://labios-argentos-vsr-clean-v1/argentina/new_discovery/rois_npz/EhYznjqlcKY/clip_0007.npz key=rois shape=151x96x96 dtype=uint8

## clean_gpt_v1 TXT samples


## large/turbo TXT samples

- large gs://labios-argentos-vsr-clean-v1/argentina/existing/transcripts/large/24HS EN LOS HOTELES MÁS RAROS DE ARGENTINA/clip_0014.txt chars=161 sample='es francis tachara la de coger un avion vieron que ahi esta ese mito de que hay cosas que tenes que hacer antes de morir'
- large gs://labios-argentos-vsr-clean-v1/argentina/existing/transcripts/large/24HS EN LOS HOTELES MÁS RAROS DE ARGENTINA/clip_0029.txt chars=49 sample='chabona puso un huevo mientras estabamos grabando'
- large gs://labios-argentos-vsr-clean-v1/argentina/existing/transcripts/large/24HS EN LOS HOTELES MÁS RAROS DE ARGENTINA/clip_0034.txt chars=157 sample='nada que ver es un lugar muy barato para la experiencia que te dan yo creo que en el video les dije cuanto le salia la n'
- large gs://labios-argentos-vsr-clean-v1/argentina/existing/transcripts/large/24HS EN LOS HOTELES MÁS RAROS DE ARGENTINA/clip_0035.txt chars=61 sample='no no era como 200 mil pesos dos personas una cosa asi boludo'
- large gs://labios-argentos-vsr-clean-v1/argentina/existing/transcripts/large/24HS EN LOS HOTELES MÁS RAROS DE ARGENTINA/clip_0038.txt chars=135 sample='distinto el jugo es exprimido de natural porque recien lo vi al hombre como cortada a la naranja asi que pues tiene una '
- turbo gs://labios-argentos-vsr-clean-v1/argentina/existing/transcripts/turbo/f01__AN_CDOTA_VIAJE_MUNDIAL_BRASIL_2014_parte_1/clip_0001.txt chars=190 sample='contexto argentina selección 2014 el maestro sabela al frente de la selección argentina parecía que no dábamos ni dos ma'
- turbo gs://labios-argentos-vsr-clean-v1/argentina/existing/transcripts/turbo/f01__AN_CDOTA_VIAJE_MUNDIAL_BRASIL_2014_parte_1/clip_0002.txt chars=205 sample='No salva Messi. Cuestión. Yo voy a ser sincero. Yo tampoco tenía tanta fe en la selección. Lo bancaba a Sabela. Pero hab'
- turbo gs://labios-argentos-vsr-clean-v1/argentina/existing/transcripts/turbo/f01__AN_CDOTA_VIAJE_MUNDIAL_BRASIL_2014_parte_1/clip_0003.txt chars=74 sample='con Holanda, nos juntábamos todos los Vader a ver el partido, estábamos en'
- turbo gs://labios-argentos-vsr-clean-v1/argentina/existing/transcripts/turbo/f01__AN_CDOTA_VIAJE_MUNDIAL_BRASIL_2014_parte_1/clip_0006.txt chars=122 sample='discutíamos. Entonces, contra Holanda me dice, hagamos una cosa si le ganamos a Holanda nos vamos a Brasil, a ver la fin'
- turbo gs://labios-argentos-vsr-clean-v1/argentina/existing/transcripts/turbo/f01__AN_CDOTA_VIAJE_MUNDIAL_BRASIL_2014_parte_1/clip_0007.txt chars=216 sample='Entonces yo agarro y digo, sí, porque, o sea, yo no creía que le vayamos a ganar a Holanda, acuérdense que ese Holanda f'

## New discovery large/turbo TXT samples

- large gs://labios-argentos-vsr-clean-v1/argentina/new_discovery/transcripts/large/EhYznjqlcKY/clip_0000.txt chars=138 sample='¿Cuándo me voy a dedicar a la economía? Es la pregunta natural en un programa que se llama Momento Económico que hace 14'
- large gs://labios-argentos-vsr-clean-v1/argentina/new_discovery/transcripts/large/EhYznjqlcKY/clip_0001.txt chars=109 sample='toda la emisión a discutir el último libro de Emilio Perina y la semana pasada a discutir con la doclización.'
- large gs://labios-argentos-vsr-clean-v1/argentina/new_discovery/transcripts/large/EhYznjqlcKY/clip_0002.txt chars=103 sample='etcétera, pensaba dedicarme a la economía pero las circunstancias que hicieron no muestre a mi invitado'
- large gs://labios-argentos-vsr-clean-v1/argentina/new_discovery/transcripts/large/EhYznjqlcKY/clip_0003.txt chars=113 sample='por el momento. Quizás hablemos de economía, quizás no, realmente todavía no lo sé, pero me van un par de minutos'
- large gs://labios-argentos-vsr-clean-v1/argentina/new_discovery/transcripts/large/EhYznjqlcKY/clip_0004.txt chars=104 sample='La realidad me ha ido acumulando algunas cosas que quería en las últimas dos semanas, no las pude decir.'
- turbo gs://labios-argentos-vsr-clean-v1/argentina/new_discovery/transcripts/turbo/EhYznjqlcKY/clip_0000.txt chars=133 sample='¿Cuándo me voy a dedicar a la economía? Es la pregunta rural en un programa que se llama Momento Económico que hace 14 d'
- turbo gs://labios-argentos-vsr-clean-v1/argentina/new_discovery/transcripts/turbo/EhYznjqlcKY/clip_0001.txt chars=111 sample='toda la emisión a discutir el último libro de Emilio Perina y la semana pasada a discutir con la documentación.'
- turbo gs://labios-argentos-vsr-clean-v1/argentina/new_discovery/transcripts/turbo/EhYznjqlcKY/clip_0002.txt chars=89 sample='pensaba dedicar la economía pero las circunstancias que hicieron no muestre a mi invitado'
- turbo gs://labios-argentos-vsr-clean-v1/argentina/new_discovery/transcripts/turbo/EhYznjqlcKY/clip_0003.txt chars=122 sample='por el momento. Quizás hablemos de economía, quizás no, realmente todavía no lo sé, pero me van a tomar un par de minuto'
- turbo gs://labios-argentos-vsr-clean-v1/argentina/new_discovery/transcripts/turbo/EhYznjqlcKY/clip_0004.txt chars=104 sample='La realidad me ha ido acumulando algunas cosas que quería en las últimas dos semanas, no las pude decir.'
