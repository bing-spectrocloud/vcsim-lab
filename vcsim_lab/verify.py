"""Verify the live lab against vms.yaml.

Core checks (existence, CPU, memory, guest, firmware, disks, NICs, boot
order, power, video 3D, template flag, vAppConfig) must match. TPM, GPU
and Secure Boot are best-effort: vcsim accepts the vTPM and GPU devices
but serves some of their detail back degraded, and it drops the
efiSecureBootEnabled flag entirely even though the flag is sent (verified
against govmomi v0.56.2), so a mismatch there is a warning, not a
failure — the report says which. Templates and vAppConfig are core:
vcsim serves config.template and the full vAppConfig back reliably.

The per-VM attributes are checked against the resolved assignment
(seed-resolved, see config.resolve_random_assignments): CBT, power
state, boot delay and disk.EnableUUID are core (vcsim serves them
reliably); tools status is best-effort — it is guest-reported with
no setter API and vcsim always reports toolsNotInstalled, so a
mismatch is a warning.

The basic/dynamic disk layout is declared-only (no vSphere property),
but VMs with dynamic disks carry the layout in extraConfig, so verify
checks the live extraConfig value against the declared layout when
present, and warns when the live VM carries no layout at all.

Snapshots are a core check: vcsim creates them and serves the tree
back (names included), so the live snapshot names must match the
declared list. When the live summary carries no snapshot state at all
(None — the leaf retrieve failed), the check is skipped rather than
failed, the same way CBT and boot delay skip on missing live values.

The folder path is a core check too: vcsim creates nested folders and
serves the parent chain back, so the live path (relative to the
datacenter vm folder, "" for a VM directly under it) must match the
declared `folder:` path. Skipped when the live summary has no folder
value (None — the parent chain could not be read).
"""

from __future__ import annotations

import json

from . import devices
from .config import (
    POWER_STATES,
    TOOLS_STATUSES,
    LabConfig,
    ResolvedAssignment,
    VCenterConfig,
    VmSpec,
    disk_layout_extra_config,
    resolve_random_assignments,
)
from .query import summarize_lab
from .vcenter import connect, disconnect


def _check(label: str, expected, actual, failures: list, warnings: list, best_effort: bool = False):
    if expected == actual:
        return
    msg = f"{label}: expected {expected!r}, live is {actual!r}"
    (warnings if best_effort else failures).append(msg)


def _default_assignment(spec: VmSpec) -> ResolvedAssignment:
    """Fallback when no resolved assignment is given (offline tests):
    explicit values pass through; "random" fields take the neutral
    default (CBT off, standard boot delay, tools not installed)."""
    if spec.template:
        power_state = "poweredOff"
    elif spec.power_state in POWER_STATES:
        power_state = spec.power_state
    else:
        power_state = "poweredOn" if spec.power_on else "poweredOff"
    return ResolvedAssignment(
        cbt=spec.cbt is True,
        power_state=power_state,
        fast_boot=spec.fast_boot is True,
        tools_status=(
            spec.tools_status if spec.tools_status in TOOLS_STATUSES else "toolsNotInstalled"
        ),
        disk_enable_uuid=spec.disk_enable_uuid is True,
    )


def verify_vm(
    spec: VmSpec,
    live: dict | None,
    assigned: ResolvedAssignment | None = None,
) -> tuple[list[str], list[str]]:
    failures, warnings = [], []
    if live is None:
        return [f"VM {spec.name!r} does not exist"], []
    assigned = assigned or _default_assignment(spec)
    _check("guest_id", spec.guest_id, live["guest_id"], failures, warnings)
    _check("cpus", spec.cpus, live["cpus"], failures, warnings)
    _check("memory_mb", spec.memory_mb, live["memory_mb"], failures, warnings)
    _check("firmware", spec.firmware, live["firmware"], failures, warnings)
    _check("hardware_version", spec.hardware_version, live["hardware_version"], failures, warnings)
    _check("power", assigned.power_state, live["power"], failures, warnings)
    _check("boot_order", spec.boot_order, live["boot_order"], failures, warnings)
    _check("template", spec.template, live.get("template", False), failures, warnings)
    # Folder path is core: vcsim creates nested folders and serves the
    # parent chain back. Skipped when the live summary has no folder
    # value (None — the chain could not be read, not "root").
    if live.get("folder") is not None:
        _check("folder", spec.folder, live["folder"], failures, warnings)
    # CBT and boot delay are core: vcsim serves both reliably. Checks
    # skip when the live dict has no value (offline test fixtures).
    if live.get("cbt") is not None:
        _check("cbt", assigned.cbt, live["cbt"], failures, warnings)
    if live.get("boot_delay_ms") is not None:
        _check(
            "boot_delay_ms",
            assigned.expected_boot_delay_ms(),
            live["boot_delay_ms"],
            failures,
            warnings,
        )
    # disk.EnableUUID is core: the builder stamps it into extraConfig
    # on every VM and vcsim serves extraConfig back intact. Skipped
    # when the live summary has no value (None — a VM built before the
    # feature existed, where the key is absent).
    if live.get("disk_enable_uuid") is not None:
        _check(
            "disk_enable_uuid",
            assigned.disk_enable_uuid,
            live["disk_enable_uuid"],
            failures,
            warnings,
        )
    if spec.vapp is not None:
        live_vapp = live.get("vapp") or {}
        product = live_vapp.get("product") or {}
        _check(
            "vapp product",
            (spec.vapp.name, spec.vapp.vendor, spec.vapp.version),
            (product.get("name"), product.get("vendor"), product.get("version")),
            failures,
            warnings,
        )
        _check(
            "vapp properties",
            {p.id: p.value for p in spec.vapp.properties},
            {p["id"]: p["value"] for p in live_vapp.get("properties", [])},
            failures,
            warnings,
        )

    # Disks are compared as a multiset of full tuples, so a disk's
    # attributes stay paired with its size/datastore. RDM disks have
    # no thin provisioning and no diskMode (vcsim serves None for
    # both), so the expected values are None there. Live dicts from
    # older fixtures may lack the newer keys — .get() defaults match
    # the declared defaults, so those fixtures still pass.
    def _expected_disk(d):
        if d.backing == "rdm":
            return (d.size_gb, d.datastore, None, None, "rdm", False, d.rdm_compatibility)
        return (d.size_gb, d.datastore, d.thin, d.disk_mode, "vmdk", d.shared, None)

    def _live_disk(d):
        return (
            d["capacity_gb"],
            d["datastore"],
            d["thin"],
            d.get("disk_mode", "persistent"),
            d.get("backing", "vmdk"),
            d.get("shared", False),
            d.get("rdm_compatibility"),
        )

    def _disk_sort_key(t):
        # Tuples mix None (RDM disks have no thin/diskMode) with bools
        # and strings, which Python cannot compare directly — wrap
        # each element so None sorts first within its position.
        return tuple((0, "") if v is None else (1, v) for v in t)

    expected_disks = sorted((_expected_disk(d) for d in spec.disks), key=_disk_sort_key)
    actual_disks = sorted((_live_disk(d) for d in live["disks"]), key=_disk_sort_key)
    _check(
        "disks (size_gb, datastore, thin, disk_mode, backing, shared, rdm_compatibility)",
        expected_disks,
        actual_disks,
        failures,
        warnings,
    )
    # Disk layout (basic/dynamic): declared-only — vSphere has no
    # property for it — but VMs with dynamic disks carry the layout in
    # extraConfig (lab.diskLayout), so a present-but-wrong live value
    # is a failure while a missing one (VM built before the feature, or
    # extraConfig not served) is only a warning.
    expected_layout = disk_layout_extra_config(spec)
    if expected_layout is not None:
        live_layout = live.get("disk_layout")
        if live_layout is None:
            warnings.append(
                f"disk_layout: declared {expected_layout!r} but the live VM "
                "carries no lab.diskLayout extraConfig (declared-only)"
            )
        else:
            _check("disk_layout", expected_layout, live_layout, failures, warnings)
    _check("nic count", len(spec.nics), len(live["nics"]), failures, warnings)
    # NIC device types (emulated card class, or VirtualSriovEthernetCard
    # for SR-IOV). Skipped when the live summaries carry no type (older
    # fixtures only record the label).
    if live["nics"] and all("type" in n for n in live["nics"]):
        _check(
            "nic types",
            sorted(devices.nic_device_class_name(n) for n in spec.nics),
            sorted(n["type"] for n in live["nics"]),
            failures,
            warnings,
        )
    _check("cdrom", spec.cdrom is not None, live["cdrom"], failures, warnings)
    _check("video_3d", spec.video_3d, live["video_3d"], failures, warnings)
    # Snapshots are core: vcsim creates them and serves the tree back
    # (names included). Skipped when the live summary has no snapshot
    # state (None — the leaf retrieve failed, not "zero snapshots").
    if live.get("snapshots") is not None:
        _check(
            "snapshots",
            [s.name for s in spec.snapshots],
            [s["name"] for s in live["snapshots"]],
            failures,
            warnings,
        )

    # Best-effort: the simulator may legitimately refuse these.
    _check("tpm", spec.tpm, live["tpm"], failures, warnings, best_effort=True)
    _check("secure_boot", spec.secure_boot, live["secure_boot"], failures, warnings, best_effort=True)
    _check(
        "gpu present",
        spec.gpu is not None,
        len(live["gpus"]) > 0,
        failures,
        warnings,
        best_effort=True,
    )
    # Tools status is guest-reported with no setter API; vcsim always
    # reports toolsNotInstalled, so a mismatch is a warning, not a
    # failure — the declared value is what the lab *expects*, not what
    # the simulator can serve.
    if live.get("tools_status") is not None:
        _check(
            "tools_status",
            assigned.tools_status,
            live["tools_status"],
            failures,
            warnings,
            best_effort=True,
        )
    return failures, warnings


def verify_lab(cfg: VCenterConfig, lab: LabConfig) -> dict:
    si = connect(cfg)
    try:
        live_by_name = {s["name"]: s for s in summarize_lab(si)}
    finally:
        disconnect(si)
    assignments = resolve_random_assignments(lab)
    result = {"vms": {}, "failures": 0, "warnings": 0}
    for spec in lab.vms:
        failures, warnings = verify_vm(spec, live_by_name.get(spec.name), assignments[spec.name])
        result["vms"][spec.name] = {"failures": failures, "warnings": warnings}
        result["failures"] += len(failures)
        result["warnings"] += len(warnings)
    return result


def print_result(result: dict) -> bool:
    ok = result["failures"] == 0
    for name, checks in result["vms"].items():
        failures, warnings = checks["failures"], checks["warnings"]
        if not failures and not warnings:
            print(f"{name}: OK")
            continue
        print(f"{name}:")
        for msg in failures:
            print(f"  FAIL {msg}")
        for msg in warnings:
            print(f"  WARN {msg} (best-effort: vcsim may not support it)")
    print()
    print(f"{len(result['vms'])} VMs checked, {result['failures']} failures, {result['warnings']} warnings")
    return ok


def result_json(result: dict) -> str:
    return json.dumps(result, indent=2)
