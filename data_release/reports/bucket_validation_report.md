# Bucket validation report

bucket: gs://labios-argentos-vsr-clean-v1/
iam_fg_object_viewer: true
resource_filter: (name~vsr-full-clean OR name~vsr-cleaning-vm OR labels.task=full-clean-release)
remaining_vms_matching_filter: ""
remaining_disks_matching_filter: ""
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

## Manifest rows

- argentina_existing_manifest_rows: 9191
- argentina_new_manifest_rows: 20
- spanish_general_manifest_rows: 47152
- clean_manifest_rows: 9191
- alignment_manifest_rows: 9191
- existing_reconstruction_manifest_rows: 2193
- asr_large_turbo_manifest_rows: 4392
- final_release_manifest_rows: 9191
- final_train_manifest_rows: 8067
- final_eval_manifest_rows: 1124
- new_discovery_ingest_manifest_rows: 20
- spanish_general_asr_manifest_rows: 47152

## MP4 samples

- gs://labios-argentos-vsr-clean-v1/argentina/existing/clips_mp4/24HS EN LOS HOTELES MÁS RAROS DE ARGENTINA/clip_0014.mp4 video=true audio=false audio_zero_marker=true
- gs://labios-argentos-vsr-clean-v1/argentina/existing/clips_mp4/24HS EN LOS HOTELES MÁS RAROS DE ARGENTINA/clip_0029.mp4 video=true audio=false audio_zero_marker=true
- gs://labios-argentos-vsr-clean-v1/argentina/existing/clips_mp4/24HS EN LOS HOTELES MÁS RAROS DE ARGENTINA/clip_0034.mp4 video=true audio=false audio_zero_marker=true
- gs://labios-argentos-vsr-clean-v1/argentina/existing/clips_mp4/24HS EN LOS HOTELES MÁS RAROS DE ARGENTINA/clip_0035.mp4 video=true audio=false audio_zero_marker=true
- gs://labios-argentos-vsr-clean-v1/argentina/existing/clips_mp4/24HS EN LOS HOTELES MÁS RAROS DE ARGENTINA/clip_0038.mp4 video=true audio=false audio_zero_marker=true

## Reconstructed MP4 samples

- gs://labios-argentos-vsr-clean-v1/argentina/existing/clips_with_audio/f01__AN_CDOTA_VIAJE_MUNDIAL_BRASIL_2014_parte_1/clip_0001.mp4 video=true audio=true audio_zero_marker=false
- gs://labios-argentos-vsr-clean-v1/argentina/existing/clips_with_audio/f01__AN_CDOTA_VIAJE_MUNDIAL_BRASIL_2014_parte_1/clip_0002.mp4 video=true audio=true audio_zero_marker=false
- gs://labios-argentos-vsr-clean-v1/argentina/existing/clips_with_audio/f01__AN_CDOTA_VIAJE_MUNDIAL_BRASIL_2014_parte_1/clip_0003.mp4 video=true audio=true audio_zero_marker=false
- gs://labios-argentos-vsr-clean-v1/argentina/existing/clips_with_audio/f01__AN_CDOTA_VIAJE_MUNDIAL_BRASIL_2014_parte_1/clip_0006.mp4 video=true audio=true audio_zero_marker=false
- gs://labios-argentos-vsr-clean-v1/argentina/existing/clips_with_audio/f01__AN_CDOTA_VIAJE_MUNDIAL_BRASIL_2014_parte_1/clip_0007.mp4 video=true audio=true audio_zero_marker=false

## NPZ samples

- gs://labios-argentos-vsr-clean-v1/argentina/existing/rois_npz/24HS EN LOS HOTELES MÁS RAROS DE ARGENTINA/clip_0014.npz key=rois shape=255x96x96 dtype=uint8
- gs://labios-argentos-vsr-clean-v1/argentina/existing/rois_npz/24HS EN LOS HOTELES MÁS RAROS DE ARGENTINA/clip_0029.npz key=rois shape=85x96x96 dtype=uint8
- gs://labios-argentos-vsr-clean-v1/argentina/existing/rois_npz/24HS EN LOS HOTELES MÁS RAROS DE ARGENTINA/clip_0034.npz key=rois shape=201x96x96 dtype=uint8
- gs://labios-argentos-vsr-clean-v1/argentina/existing/rois_npz/24HS EN LOS HOTELES MÁS RAROS DE ARGENTINA/clip_0035.npz key=rois shape=107x96x96 dtype=uint8
- gs://labios-argentos-vsr-clean-v1/argentina/existing/rois_npz/24HS EN LOS HOTELES MÁS RAROS DE ARGENTINA/clip_0038.npz key=rois shape=185x96x96 dtype=uint8

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
