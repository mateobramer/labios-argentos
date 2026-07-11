# Human review pack

Este pack permite destrabar fuentes sin tocar columnas automaticas.

## Archivos

- `human_source_mapping_needed.csv`: completar `manual_url`, `manual_confidence` y `manual_notes` para fuentes existing.
- `human_new_discovery_download_needed.csv`: completar `manual_alternative_url` o `manual_download_path` para accepted new_discovery que no descargan automaticamente.

## Reglas

- No editar columnas automaticas.
- `manual_confidence` debe ser `high`, `medium` o `low`.
- `manual_download_path` debe apuntar a un archivo local descargado fuera del repo o en un directorio ignorado.
- No guardar cookies, tokens ni credenciales en este directorio.

## Como reanudar despues de completar el CSV

1. Descargar fuentes localmente con:

```powershell
python data_release/scripts/download_sources_local.py --new-discovery --limit 5
```

2. Para existing con `manual_url`, reintentar reconstruccion desde una fuente ya descargada/subida a GCS.

```powershell
python data_release/scripts/download_sources_local.py --existing --limit 5
```

3. Crear VM GPU solo cuando haya fuentes en GCS listas para segmentar/ASR.

## Top existing a revisar

- f07__DAVOO_XENEIZE_OPINA_DE_BOCA_0_UNIVERSIDAD_CATOLICA clips=937 reason=blocked_alignment_failed=23|baseline_existing_only=128|low_alignment_score_present
- f13__JULI_POGGIO_EN_FERN_CON_GREGO clips=556 reason=blocked_alignment_failed=5|baseline_existing_only=551|low_alignment_score_present|source_reconstruction_failed
- f22__ME_ACUSARON_DE_BRUJA_Y_ME_TUVE_QUE_IR_DEL_PUEBLO_- clips=542 reason=blocked_alignment_failed=6|baseline_existing_only=536|low_alignment_score_present|source_reconstruction_failed
- f42__Por_qu_ya_no_podemos_conversar_y_c_mo_aprender_a clips=373 reason=blocked_alignment_failed=5|baseline_existing_only=368|low_alignment_score_present|source_reconstruction_failed
- f12__GASPI_DAMIAN_KUC_LA_REINI_BOTTERO_EL_MUNDIAL_Y clips=360 reason=blocked_alignment_failed=7|baseline_existing_only=32|low_alignment_score_present
- f27__MICA_SUAREZ_KEVSHO_LULI_GONZALEZ_JAMES_CHARLES clips=331 reason=blocked_alignment_failed=5|baseline_existing_only=326|low_alignment_score_present|source_reconstruction_failed
- f01__AN_CDOTA_VIAJE_MUNDIAL_BRASIL_2014_parte_1 clips=302 reason=blocked_alignment_failed=23|baseline_existing_only=89|low_alignment_score_present
- f11__Entrevista_completa_por_mi_libro_Franco_con_Diego clips=295 reason=blocked_alignment_failed=1|baseline_existing_only=13|low_alignment_score_present
- f29__Manipulaci_n_mental_C_mo_tu_CEREBRO_literalmente_c clips=291 reason=match_confidence=low|blocked_source_not_found=291
- f18__Las_y_los_estudiantes_al_frente_-_Entrevista_a_la clips=257 reason=blocked_alignment_failed=124|baseline_existing_only=133|low_alignment_score_present
- f05__CHARLA_SOBRE_EL_AMOR_Y_EL_DESAMOR clips=233 reason=blocked_alignment_failed=96|baseline_existing_only=26|low_alignment_score_present
- f28__MILITANDO_el_AJUSTE_ESTA_SEMANA_EN_SPRINGFIELD_co clips=223 reason=match_confidence=none|blocked_source_not_found=223
- f03__AZZARO_REACCI_N_-_RIVER_A_LA_FINAL_LE_GAN_1-0_A clips=221 reason=blocked_alignment_failed=6|baseline_existing_only=20|low_alignment_score_present
- f09__ESTOY_EN_UN_BROTE_PERDON clips=207 reason=blocked_alignment_failed=1|baseline_existing_only=3|low_alignment_score_present|spot_vm_terminated
- f49__RIVER_GAN_SUFRIENDO_VS_CIUDAD_DE_BOLIVAR_ALIVI clips=199 reason=baseline_existing_only=199|low_alignment_score_present
- f06__Coronavirus_conferencia_completa_de_Alberto_Fern_n clips=184 reason=blocked_alignment_failed=184|low_alignment_score_present
- f54__TODOS_ODIAMOS_A_MILEI_PAULA_CHAVEZ_ESTA_PODRIDA clips=184 reason=match_confidence=low|blocked_source_not_found=184
- f53__TODO_el_COLEGIO_me_HAC_A_BULLYING..._StoryTime_C clips=180 reason=blocked_alignment_failed=1|baseline_existing_only=179|low_alignment_score_present
- f02__AZZARO_REACCI_N_-_CICLO_TERMINADO_RACING_EMPAT_2- clips=165 reason=match_confidence=low|blocked_source_not_found=165
- f48__RESPONDO_TODO clips=158 reason=match_confidence=none|blocked_source_not_found=158

## Top new_discovery a descargar

- ITovsJg-q5c clips=1341.0 Cómo Amarse a uno Mismo y Tener Buena Salud Mental: Hábitos de Vida - Gabriel Rolón
- h3HtBhArO1Q clips=1228.0 El Método Rebord #56 - Andy Chango
- OrwtPwftIi4 clips=1023.0 🎤 Curso de Oratoria con Daniel Colombo | 100% Práctico
- qEKPgqURvo0 clips=822.0 REBORD - PERETTI | HAY ALGO AHÍ | BLENDER
- jgp8WZvtkWU clips=748.0 Martin Menem con Iván Schargrodsky en #OnTheRecord
- YYIVFA000BI clips=682.0 Pepe Mujica con Jorge Fontevecchia (Entrevista Completa)
- WMk6afYRfKM clips=680.0 Entrevista en “IP Noticias” con Noelia Barral, Romina Calderaro y Nora Veiras - IP - 07/02/2021
- j4x2GC1Ztro clips=677.0 Mario Pergolini y un imperdible mano a mano con Andy Kusnetzoff | #Perros2022 Perros de la Calle
- eqw0QM4A0oA clips=598.0 Taty Almeida | Bios Militantes con Julia Mengolini en #Segurola
- Z-t-GNlxpYc clips=596.0 Entrevista completa de Javier Milei con Luis Majul: "Hemos pasado el momento bisagra"
- JixCyhEGE0A clips=453.0 Javier Milei en TN I Entrevista completa del 30/06/2024
- EhYznjqlcKY clips=408.0 Guido Di Tella con Juan Carlos de Pablo - DiFilm (1998)
- IGYG0Kn0wxo clips=374.0 ENTREVISTA: DIEGO TORRES
- PRQEkiIWps0 clips=331.0 Coscu Mete Un Bombazo Histórico Después De Probar Las Salsas Más Picantes Del Mundo
- EuSM3LscaWI clips=306.0 Marcos Galperin | La vida del emprendedor | Aprender de Grandes #072
- q94HfK07DjI clips=295.0 El último reportaje del papa Francisco con Infobae
- R3f0x1IJvhI clips=256.0 GUILLERMO FRANCELLA, UN LUJO COMO PRIMER INVITADO DE OTRO DÍA PERDIDO || ENTREVISTA COMPLETA
- yJxDKBgw5NU clips=248.0 Santiago Cafiero, mano a mano en TN: "Es una decisión sanitaria, no política"
- vDNy6EN7bIY clips=156.0 ¿Cómo es el ESPAÑOL ARGENTINO FORMAL? - Argento Podcast #44
- sjnH4bTak9s clips=110.0 Obstáculos ideológicos en el empleo
