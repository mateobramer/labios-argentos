# Cost/runtime report

project: labios-argentos-499900
bucket_destino: gs://labios-argentos-vsr-clean-v1/

## VM temporal

- vm_name: vsr-cleaning-vm-20260706-2321-l4-uscentral1c
- zone: us-central1-c
- machine_type: g2-standard-8
- gpu: NVIDIA L4
- gpu_memory: 23034 MiB
- driver_version: 580.159.03
- provisioning_model: SPOT
- boot_disk: 200 GB pd-balanced, auto-delete true
- external_ip: ephemeral
- final_status: deleted

## Intentos VM

- Intento inicial `common-cu121`: fallo porque la image family ya no existe.
- Intentos L4 en `us-east1-b/c/d` y `us-central1-b`: stockout/resource availability.
- Exito: Spot L4 en `us-central1-c`.

## Limpieza recursos

Verificado al final:

- remaining_vms_named_vsr_cleaning: none
- remaining_disks_named_vsr_cleaning: none
- remaining_static_addresses_named_vsr_cleaning: none

## ASR

No se corrio ASR masivo en GPU porque los `.mp4` existentes del bucket son ROIs
sin audio. La VM se uso solo para validar capacidad GPU (`nvidia-smi`) y se borro
inmediatamente para evitar costo.

Para escalar ASR a `turbo`, hace falta reconstruir clips con audio desde URLs o
raw videos fuente.
