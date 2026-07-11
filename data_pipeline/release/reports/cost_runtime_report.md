# Cost/runtime report

project: labios-argentos-499900
vm_runs:
- name: vsr-full-clean-20260707-0200; zone: us-central1-a; machine_type: g2-standard-8; gpu: nvidia-l4; provisioning_model: SPOT; status: used_then_spot_terminated_then_deleted
- name: vsr-full-clean-continue-20260707; zone: us-east1-d; machine_type: g2-standard-8; gpu: nvidia-l4; provisioning_model: STANDARD; status: used_for_resume_then_deleted
outputs_synced_to_gcs: true
cleanup_verification: data_pipeline/release/reports/bucket_validation_report.md shows no matching instances/disks/static IPs
