"""Query the lab: the pyVmomi read path, mirroring what the builder wrote.

``summarize_vm`` is the reference implementation for "give me everything
about a VM" — OS, CPU/memory, firmware, TPM, GPU, boot order, disks and
NICs — using plain attribute access on the objects a ContainerView
returns. For large inventories the same fields can be pulled through a
PropertyCollector (see ``summarize_vm``'s notes), but for a lab of this
size the direct read is clearer and fast enough.

The per-VM attributes (CBT, power state, boot delay / fast boot, tools
status) come from leaf property retrieves where the whole object cannot
be deserialized against vcsim; ``summarize_lab`` accepts the resolved
assignments so the summary can show declared-vs-live values.
"""

from __future__ import annotations

import json

from pyVmomi import vim

from . import devices
from .config import DISK_ENABLE_UUID_EXTRA_CONFIG_KEY, DISK_LAYOUT_EXTRA_CONFIG_KEY
from .vcenter import (
    connect,
    disconnect,
    folder_rel_path,
    list_objs,
    retrieve_props,
    retrieve_props_multi,
    safe_name,
)


def _disk_label_number(device) -> int:
    """The N in a disk's "Hard disk N" label, for ordering disks the
    way the spec lists them (live devices are not guaranteed to come
    back in creation order). Unparseable labels sort last."""
    label = getattr(device.deviceInfo, "label", "") or ""
    try:
        return int(label.rsplit(" ", 1)[1])
    except (ValueError, IndexError):
        return 1 << 30


def _isoformat(value):
    """A datetime (or anything with .isoformat) as an ISO string, else
    str(value) — snapshot createTime arrives as a datetime from
    pyVmomi, but the helper stays tolerant of plain strings."""
    if value is None:
        return None
    iso = getattr(value, "isoformat", None)
    return iso() if callable(iso) else str(value)


def _snapshot_tree_entries(trees) -> tuple[list[dict], dict[str, str]]:
    """Flatten a snapshot tree (pre-order) into plain dicts.

    Returns (entries, moid -> name): the second map resolves the
    currentSnapshot MoRef to a name without another round trip. Each
    entry carries the snapshot's name, description, numeric id, power
    state at capture time and creation time. A VM with no snapshots
    has an empty rootSnapshotList and yields ([], {}).
    """
    entries: list[dict] = []
    by_moid: dict[str, str] = {}

    def _walk(nodes):
        for node in nodes or []:
            snap_ref = getattr(node, "snapshot", None)
            moid = getattr(snap_ref, "_moId", None) if snap_ref is not None else None
            if moid is not None:
                by_moid[moid] = getattr(node, "name", None)
            state = getattr(node, "state", None)
            entries.append(
                {
                    "name": getattr(node, "name", None),
                    "description": getattr(node, "description", "") or "",
                    "id": getattr(node, "id", None),
                    "state": str(state) if state is not None else None,
                    "create_time": _isoformat(getattr(node, "createTime", None)),
                }
            )
            _walk(getattr(node, "childSnapshotList", None))

    _walk(trees)
    return entries, by_moid


def _retrieve_snapshot_props(si, vms) -> dict:
    """Best-effort snapshot state for each VM.

    Returns {moId: {"snapshots": [...], "current_snapshot": name |
    None}}. Snapshot state comes from leaf retrieves
    (``snapshot.rootSnapshotList``, ``snapshot.currentSnapshot``) —
    the whole ``vm.snapshot`` object is not read directly, for the
    same deserialization reasons as ``vm.runtime`` (see
    ``vcenter.retrieve_props``). A VM whose snapshot state cannot be
    read at all gets ``"snapshots": None`` (unknown, not zero) so
    verify skips the check instead of failing it; a VM with no
    snapshots gets an empty list. A batched failure (e.g. one VM's
    tree not deserializing) falls back to per-VM retrieves so one bad
    VM does not poison the rest.
    """
    paths = ["snapshot.rootSnapshotList", "snapshot.currentSnapshot"]
    try:
        leaf = retrieve_props_multi(si, vms, vim.VirtualMachine, paths)
    except Exception:  # noqa: BLE001 — snapshot state must not break the summary
        leaf = {}
        for vm in vms:
            try:
                leaf[vm._moId] = retrieve_props(si, vm, vim.VirtualMachine, paths)
            except Exception:  # noqa: BLE001 — per-VM best effort
                continue
    out = {}
    for vm in vms:
        props = leaf.get(vm._moId)
        if props is None:
            out[vm._moId] = {"snapshots": None, "current_snapshot": None}
            continue
        entries, by_moid = _snapshot_tree_entries(props.get("snapshot.rootSnapshotList"))
        current = props.get("snapshot.currentSnapshot")
        current_name = by_moid.get(getattr(current, "_moId", None)) if current is not None else None
        out[vm._moId] = {"snapshots": entries, "current_snapshot": current_name}
    return out


# ---------------------------------------------------------------------------
# VM summaries
# ---------------------------------------------------------------------------


def _gpu_info(device) -> dict:
    backing = device.backing
    info = {"type": type(device).__name__}
    if isinstance(backing, vim.vm.device.VirtualPCIPassthrough.DeviceBackingInfo):
        info.update(
            {
                "mode": "passthrough",
                "id": backing.id,
                "device_id": backing.deviceId,
                "system_id": backing.systemId,
                "vendor_id": backing.vendorId,
            }
        )
    elif isinstance(backing, vim.vm.device.VirtualPCIPassthrough.DynamicBackingInfo):
        allowed = [
            {"vendor_id": a.vendorId, "device_id": a.deviceId} for a in (backing.allowedDevice or [])
        ]
        info.update({"mode": "dynamic", "allowed_devices": allowed, "assigned_id": backing.assignedId})
    elif isinstance(backing, vim.vm.device.VirtualPCIPassthrough.VmiopBackingInfo):
        info.update({"mode": "vgpu", "profile": backing.vgpu})
    else:
        # vcsim serves some backings (notably DynamicBackingInfo) back as
        # the base DeviceBackingInfo, losing the mode detail; the PCI
        # device itself is still there, so report it as present but
        # undescribed rather than inventing a mode.
        info.update({"mode": "unknown", "detail": "backing detail not served by vcsim"})
    return info


def _gpu_short(gpu: dict) -> str:
    """One-line GPU description for the table, e.g. ``vgpu:nvidia_a100-40c``."""
    mode = gpu.get("mode") or "?"
    detail = gpu.get("profile") or gpu.get("id") or ""
    if not detail and gpu.get("allowed_devices"):
        first = gpu["allowed_devices"][0]
        detail = f"0x{first['vendor_id']:04x}:0x{first['device_id']:04x}"
    return f"{mode}:{detail}" if detail else mode


_RDM_BACKING_CLASSES = (
    vim.vm.device.VirtualDisk.RawDiskMappingVer1BackingInfo,
    vim.vm.device.VirtualDisk.RawDiskVer2BackingInfo,
)


def _disk_info(device, datastore_names: dict) -> dict:
    backing = device.backing
    ds_name = None
    file_name = getattr(backing, "fileName", "") or ""
    if file_name.startswith("["):
        ds_name = file_name[1:].split("]", 1)[0]
    # isinstance, not a __name__ comparison: pyVmomi class names are
    # fully dotted ("vim.vm.device.VirtualDisk.…"), so a short-name
    # match never fires.
    is_rdm = isinstance(backing, _RDM_BACKING_CLASSES)
    return {
        "label": device.deviceInfo.label,
        "capacity_gb": round(device.capacityInKB / 1024 / 1024, 1),
        "datastore": ds_name,
        "thin": getattr(backing, "thinProvisioned", None),
        "eagerly_scrubbed": getattr(backing, "eagerlyScrub", None),
        "disk_mode": getattr(backing, "diskMode", None),
        # RDM disks have no thin/eager provisioning and no diskMode
        # (vcsim serves None for all three); the backing type and
        # compatibility mode are what identify them.
        "backing": "rdm" if is_rdm else "vmdk",
        "rdm_compatibility": getattr(backing, "compatibilityMode", None) if is_rdm else None,
        "rdm_device_name": getattr(backing, "deviceName", None) if is_rdm else None,
        "shared": getattr(backing, "sharing", None) == "sharingMultiWriter",
        "controller_key": device.controllerKey,
        "unit_number": device.unitNumber,
        "file": file_name,
    }


def _nic_info(device) -> dict:
    backing = device.backing
    if isinstance(backing, vim.vm.device.VirtualEthernetCard.NetworkBackingInfo):
        network, backing_type = backing.network.name, "standard"
    elif isinstance(backing, vim.vm.device.VirtualEthernetCard.DistributedVirtualPortBackingInfo):
        network = backing.port.portgroupKey
        backing_type = "dvpg"
    elif isinstance(backing, vim.vm.device.VirtualEthernetCard.OpaqueNetworkBackingInfo):
        network = backing.opaqueNetworkId
        backing_type = "opaque"
    else:
        network, backing_type = None, type(backing).__name__ if backing else None
    return {
        "label": device.deviceInfo.label,
        "type": type(device).__name__,
        "network": network,
        "backing": backing_type,
        "mac": device.macAddress,
        "connected": bool(device.connectable.connected) if device.connectable else None,
        "sriov": isinstance(device, vim.vm.device.VirtualSriovEthernetCard),
    }


def summarize_vm(
    vm: vim.VirtualMachine,
    runtime: dict | None = None,
    declared: dict | None = None,
) -> dict:
    """Everything the lab tracks about one VM, as plain JSON-able data.

    ``runtime`` carries leaf properties fetched by ``summarize_lab``
    (power state, host name, CBT flag, boot delay, tools status,
    snapshots, folder path) — ``vm.runtime`` / ``vm.summary`` /
    ``vm.snapshot`` as whole objects cannot be deserialized against
    vcsim (see ``vcenter.retrieve_props``), so they are never touched
    here except for a best-effort snapshot fallback when no leaf data
    was passed. ``declared`` is the VM's resolved assignment (from
    ``config.resolve_random_assignments``), used to show
    declared-vs-live values — notably for tools status, which vcsim
    never reports (see README).
    """
    runtime = runtime or {}
    config = vm.config
    hw = config.hardware
    controllers = {}
    for dev in hw.device:
        if isinstance(dev, vim.vm.device.VirtualController):
            controllers[dev.key] = type(dev).__name__.replace("Virtual", "")

    disks = sorted(
        (d for d in hw.device if isinstance(d, vim.vm.device.VirtualDisk)),
        key=_disk_label_number,
    )
    nics = [d for d in hw.device if isinstance(d, vim.vm.device.VirtualEthernetCard)]
    video = [d for d in hw.device if isinstance(d, vim.vm.device.VirtualVideoCard)]
    tpm = devices.has_tpm(hw.device)
    gpus = [d for d in hw.device if isinstance(d, vim.vm.device.VirtualPCIPassthrough)]
    cdrom = any(isinstance(d, vim.vm.device.VirtualCdrom) for d in hw.device)

    # Declared basic/dynamic disk layout, stamped into extraConfig at
    # create time for VMs with dynamic disks (vcsim serves extraConfig
    # back intact). None for basic-only VMs and for VMs built before
    # the feature existed.
    disk_layout = None
    disk_enable_uuid = None
    for opt in config.extraConfig or []:
        if opt.key == DISK_LAYOUT_EXTRA_CONFIG_KEY:
            disk_layout = opt.value
        elif opt.key == DISK_ENABLE_UUID_EXTRA_CONFIG_KEY:
            disk_enable_uuid = str(opt.value).strip().lower() == "true"

    datastore_names = {}
    for ds in vm.datastore or []:
        datastore_names[ds._moId] = ds.name

    boot = getattr(config, "bootOptions", None)
    # CBT / boot delay: prefer the leaf retrieve, fall back to the
    # direct config read (vm.config deserializes fine against vcsim).
    cbt = runtime.get("cbt")
    if cbt is None:
        cbt = getattr(config, "changeTrackingEnabled", None)
    boot_delay_ms = runtime.get("boot_delay_ms")
    if boot_delay_ms is None and boot is not None:
        boot_delay_ms = getattr(boot, "bootDelay", None)
    try:
        guest = vm.guest
        guest_tools_status, tools_running_status, ip = (
            guest.toolsStatus,
            guest.toolsRunningStatus,
            guest.ipAddress,
        )
    except Exception:
        guest_tools_status, tools_running_status, ip = None, None, None
    tools_status = runtime.get("tools_status", guest_tools_status)
    tools_running_status = runtime.get("tools_running_status", tools_running_status)
    # Snapshots: prefer the leaf retrieve (None there means the state
    # could not be read — unknown, not zero). Without leaf data, try
    # the direct read, best effort: vm.snapshot as a whole object is
    # unreliable against vcsim, so a failure means "unknown".
    if "snapshots" in runtime:
        snapshots = runtime["snapshots"]
        current_snapshot = runtime.get("current_snapshot")
    else:
        try:
            snapshots, _ = _snapshot_tree_entries(
                getattr(vm.snapshot, "rootSnapshotList", None)
            )
        except Exception:  # noqa: BLE001 — direct snapshot read is best effort
            snapshots = None
        current_snapshot = None
    disk_entries = [_disk_info(d, datastore_names) for d in disks]
    # Attach the declared basic/dynamic type + volume to each live disk
    # (matched by 1-based index) when the lab file was loaded — the
    # type is declared-only, so it comes from vms.yaml, not the device.
    declared_layout = (declared or {}).get("disk_layout") or {}
    for entry, declared_disk in zip(disk_entries, declared_layout.get("disks") or []):
        entry["disk_type"] = declared_disk["disk_type"]
        entry["volume"] = declared_disk["volume"]
        entry["volume_kind"] = declared_disk["volume_kind"]
    summary = {
        "name": vm.name,
        "path": config.files.vmPathName,
        # Folder path relative to the datacenter vm folder
        # ("lab/Prod/Web"; "" directly under it), walked live from the
        # VM's parent chain by summarize_lab. None when the chain could
        # not be read (unknown) or when no leaf data was passed.
        "folder": runtime.get("folder"),
        "power": runtime.get("power"),
        "host": runtime.get("host"),
        "template": devices.is_template(vm),
        "vapp": devices.describe_vapp_config(vm),
        "guest_id": config.guestId,
        "guest_name": config.guestFullName,
        "annotation": config.annotation,
        "hardware_version": config.version,
        "firmware": config.firmware,
        "secure_boot": bool(boot.efiSecureBootEnabled) if boot else False,
        "cpus": config.hardware.numCPU,
        "cores_per_socket": config.hardware.numCoresPerSocket,
        "memory_mb": config.hardware.memoryMB,
        "cpu_hot_add": config.cpuHotAddEnabled,
        "memory_hot_add": config.memoryHotAddEnabled,
        "nested_virt": bool(getattr(config, "nestedHVEnabled", False)),
        "extra_devices": devices.extra_device_kinds(hw.device),
        "tpm": tpm,
        "gpus": [_gpu_info(g) for g in gpus],
        "video_3d": bool(video and video[0].enable3DSupport),
        "video_ram_kb": video[0].videoRamSizeInKB if video else None,
        "cdrom": cdrom,
        "boot_order": devices.describe_boot_order(vm),
        "boot_delay_ms": boot_delay_ms,
        "fast_boot": boot_delay_ms == 0 if boot_delay_ms is not None else None,
        "cbt": cbt,
        "controllers": sorted(set(controllers.values())),
        "disks": disk_entries,
        # Live disk-layout string from extraConfig (None for basic-only
        # VMs); the declared per-disk types live on the disk entries
        # above when the lab file was loaded.
        "disk_layout": disk_layout,
        # Live disk.EnableUUID from extraConfig (None for VMs built
        # before the feature existed — the vSphere default is FALSE).
        "disk_enable_uuid": disk_enable_uuid,
        "nics": [_nic_info(n) for n in nics],
        "tools_status": tools_status,
        "tools_running_status": tools_running_status,
        "ip": ip,
        # Snapshots: None when the state could not be read (unknown),
        # an empty list when the VM genuinely has none.
        "snapshots": snapshots,
        "snapshot_count": len(snapshots) if snapshots is not None else None,
        "current_snapshot": current_snapshot,
    }
    if declared is not None:
        summary["declared"] = declared
        # Tools status is the one attribute the builder cannot enforce
        # (guest-reported, no setter API); flag a declared/live gap
        # rather than silently showing one of them.
        summary["tools_status_mismatch"] = tools_status != declared.get("tools_status")
        if "snapshots" in declared and snapshots is not None:
            summary["snapshot_mismatch"] = [s["name"] for s in snapshots] != declared[
                "snapshots"
            ]
        if "folder" in declared and summary["folder"] is not None:
            summary["folder_mismatch"] = summary["folder"] != declared["folder"]
    return summary


def iter_vms(si) -> list[vim.VirtualMachine]:
    content = si.RetrieveContent()
    view = content.viewManager.CreateContainerView(content.rootFolder, [vim.VirtualMachine], True)
    try:
        return list(view.view)
    finally:
        view.Destroy()


def summarize_lab(si, names: list[str] | None = None, declared: dict | None = None) -> list[dict]:
    """Summaries for every VM in the inventory.

    ``declared`` maps VM name -> resolved assignment dict (from
    ``config.resolve_random_assignments``); when given, each summary
    carries its declared values plus a tools-status mismatch flag.
    """
    vms = iter_vms(si)
    # One batched leaf retrieve for every VM's power state, host, CBT
    # flag, boot delay, tools status and parent folder, plus a host
    # moId -> name map (host names come from the inventory list).
    leaf = retrieve_props_multi(
        si,
        vms,
        vim.VirtualMachine,
        [
            "runtime.powerState",
            "runtime.host",
            "config.changeTrackingEnabled",
            "config.bootOptions.bootDelay",
            "guest.toolsStatus",
            "guest.toolsRunningStatus",
            "parent",
        ],
    )
    host_names = {h._moId: h.name for h in list_objs(si.RetrieveContent(), vim.HostSystem)}
    # Snapshot state is a separate best-effort retrieve so a failure
    # there never breaks the main summary (see _retrieve_snapshot_props).
    snapshot_props = _retrieve_snapshot_props(si, vms)
    # One shared folder-path cache for all VMs: folders are shared, so
    # each parent chain is walked once (see vcenter.folder_rel_path).
    folder_cache: dict = {}
    summaries = []
    for vm in vms:
        # A VM missing from the leaf result is a stale ContainerView
        # entry (a destroyed template, still listed by vcsim) — skip it
        # rather than crashing on its first property read.
        if vm._moId not in leaf:
            continue
        props = leaf.get(vm._moId, {})
        host = props.get("runtime.host")
        snap = snapshot_props.get(vm._moId, {"snapshots": None, "current_snapshot": None})
        parent = props.get("parent")
        folder = folder_rel_path(parent, folder_cache) if parent is not None else None
        summaries.append(
            summarize_vm(
                vm,
                runtime={
                    "power": props.get("runtime.powerState"),
                    "host": host_names.get(host._moId) if host is not None else None,
                    "cbt": props.get("config.changeTrackingEnabled"),
                    "boot_delay_ms": props.get("config.bootOptions.bootDelay"),
                    "tools_status": props.get("guest.toolsStatus"),
                    "tools_running_status": props.get("guest.toolsRunningStatus"),
                    "snapshots": snap["snapshots"],
                    "current_snapshot": snap["current_snapshot"],
                    "folder": folder,
                },
                declared=(declared or {}).get(vm.name),
            )
        )
    summaries.sort(key=lambda s: s["name"])
    if names:
        wanted = set(names)
        summaries = [s for s in summaries if s["name"] in wanted]
    return summaries


# ---------------------------------------------------------------------------
# Hosts / inventory
# ---------------------------------------------------------------------------


def list_host_pci(si) -> list[dict]:
    """Hosts with their PCI devices (GPU-class first) and vGPU capability.

    This is the read side of the GPU story: what passthrough devices and
    vGPU profiles does each host actually offer?
    """
    out = []
    for host in list_objs(si.RetrieveContent(), vim.HostSystem):
        hw = host.hardware
        # vcsim does not serve HostSystem.config (graphicsInfo,
        # sharedGpuCapabilities, pciPassthruInfo); real vCenter does.
        try:
            cfg = host.config
        except Exception:
            cfg = None
        pci = []
        for dev in hw.pciDevice or []:
            pci.append(
                {
                    "id": dev.id,
                    "class": f"{dev.classId:#06x}",
                    "vendor": dev.vendorName,
                    "vendor_id": dev.vendorId,
                    "device_id": dev.deviceId,
                    "name": dev.deviceName,
                    "is_display": dev.classId in (0x0300, 0x0302),
                }
            )
        pci.sort(key=lambda d: (not d["is_display"], d["id"]))
        out.append(
            {
                "host": host.name,
                "system_uuid": hw.systemInfo.uuid,
                "model": hw.systemInfo.model,
                "pci_devices": pci,
                "graphics": (
                    [{"device_name": g.deviceName, "vendor": g.vendorName} for g in (cfg.graphicsInfo or [])]
                    if cfg is not None
                    else None
                ),
                "shared_gpu_types": list(cfg.sharedPassthruGpuTypes or []) if cfg is not None else None,
                "shared_gpu_capabilities": (
                    [
                        {"vgpu": getattr(c, "vgpu", None), "capacity": getattr(c, "capacity", None)}
                        for c in (cfg.sharedGpuCapabilities or [])
                    ]
                    if cfg is not None
                    else None
                ),
                "config_available": cfg is not None,
            }
        )
    return out


def list_guest_ids(si) -> list[dict]:
    """Guest OS identifiers the (simulated) vCenter advertises, via the
    cluster's environment browser. Use it to check a guest_id before
    adding a new OS to vms.yaml."""
    content = si.RetrieveContent()
    clusters = list_objs(content, vim.ClusterComputeResource)
    if not clusters:
        return []
    browser = clusters[0].environmentBrowser
    try:
        descriptors = browser.guestOsDescriptor or []
    except Exception:
        # vcsim does not serve EnvironmentBrowser.guestOsDescriptor.
        return []
    return [
        {"id": d.id, "family": d.family, "full_name": d.fullName} for d in descriptors
    ]


def list_inventory(si) -> dict:
    content = si.RetrieveContent()
    return {
        "datacenters": [d.name for d in list_objs(content, vim.Datacenter)],
        "clusters": [c.name for c in list_objs(content, vim.ClusterComputeResource)],
        "hosts": [h.name for h in list_objs(content, vim.HostSystem)],
        "datastores": [d.name for d in list_objs(content, vim.Datastore)],
        "networks": [
            {"name": n.name, "type": type(n).__name__} for n in list_objs(content, vim.Network)
        ],
        "dvportgroups": [n.name for n in list_objs(content, vim.DistributedVirtualPortgroup)],
        "resource_pools": [p.name for p in list_objs(content, vim.ResourcePool)],
        "vapps": [v.name for v in list_objs(content, vim.VirtualApp)],
        # VMs use safe_name: vcsim's ContainerView keeps listing
        # destroyed templates, and reading .name on one raises
        # ManagedObjectNotFound.
        "vms": [n for n in (safe_name(v) for v in list_objs(content, vim.VirtualMachine)) if n],
    }


def list_vapps(si) -> list[dict]:
    """vApp containers with their member VMs (and the vApp's own product
    info when it is served — the builder sets it at CreateVApp time).

    A vApp is a ResourcePool subtype; its member VMs are its ``vm``
    list. Member names use safe_name for the same stale-entry reason as
    list_inventory.
    """
    content = si.RetrieveContent()
    out = []
    for vapp in list_objs(content, vim.VirtualApp):
        name = safe_name(vapp)
        if name is None:
            continue
        product = None
        try:
            products = getattr(vapp.vAppConfig, "product", None) or []
            if products:
                p = products[0]
                product = {
                    "name": p.name,
                    "vendor": p.vendor,
                    "version": p.version,
                    "full_version": p.fullVersion,
                }
        except Exception:
            product = None
        try:
            members = [n for n in (safe_name(v) for v in (vapp.vm or [])) if n]
        except Exception:
            members = []
        out.append(
            {
                "name": name,
                "parent": safe_name(vapp.parent) if getattr(vapp, "parent", None) else None,
                "product": product,
                "vms": sorted(members),
                "vm_count": len(members),
            }
        )
    out.sort(key=lambda v: v["name"])
    return out


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------


def _disk_short(d: dict) -> str:
    """One-line disk description, e.g. ``100G/LocalDS_0 thin [basic]``
    or ``500G/LocalDS_1 [dynamic:SQLData/mirrored]``. The bracketed
    part is the declared Windows disk type (basic/dynamic) plus the
    volume name/kind when the disk belongs to one; it is absent when
    the lab file was not loaded."""
    text = f"{d['capacity_gb']:g}G/{d['datastore']}{' thin' if d['thin'] else ''}"
    disk_type = d.get("disk_type")
    if disk_type is None:
        return text
    if disk_type == "dynamic" and d.get("volume"):
        return f"{text} [dynamic:{d['volume']}/{d['volume_kind']}]"
    return f"{text} [{disk_type}]"


def print_table(summaries: list[dict]) -> None:
    for s in summaries:
        gpu = ", ".join(_gpu_short(g) for g in s["gpus"]) or "—"
        disks = ", ".join(_disk_short(d) for d in s["disks"])
        nics = ", ".join(f"{n['type'].replace('Virtual', '')}@{n['network']}" for n in s["nics"])
        tag = " [template]" if s.get("template") else ""
        print(f"{s['name']}{tag} [{s['power']}] {s['guest_id']} {s['cpus']}vCPU {s['memory_mb']}MB {s['firmware']}")
        folder = s.get("folder")
        if folder is not None:
            folder_line = f"  folder: {folder or '(vm folder root)'}"
            if s.get("folder_mismatch"):
                folder_line += f" [declared {s['declared']['folder']!r}]"
            print(folder_line)
        if s.get("vapp") and s["vapp"].get("product"):
            p = s["vapp"]["product"]
            props = ", ".join(f"{x['id']}={x['value']}" for x in s["vapp"]["properties"])
            print(f"  appliance: {p['name']} {p['version']} by {p['vendor']} ({props})")
        print(f"  hw {s['hardware_version']} tpm={s['tpm']} gpu={gpu} boot={' > '.join(s['boot_order']) or '(default)'}")
        tools_line = f"tools={s['tools_status']}/{s['tools_running_status']}"
        if s.get("tools_status_mismatch"):
            tools_line += (
                f" [declared {s['declared']['tools_status']} — "
                "vcsim never reports guest tools]"
            )
        print(
            f"  cbt={s['cbt']} fast_boot={s['fast_boot']} "
            f"(bootDelay {s['boot_delay_ms']}ms) {tools_line}"
        )
        if s.get("snapshot_count"):
            names = ", ".join(x["name"] for x in s["snapshots"])
            print(f"  snapshots: {names} ({s['snapshot_count']})")
        print(f"  controllers: {', '.join(s['controllers'])}")
        print(f"  disks: {disks}")
        # When the lab file was not loaded there is no declared type on
        # the disk entries; the live extraConfig string still shows the
        # layout for dynamic-disk VMs.
        if s.get("disk_layout") and not any("disk_type" in d for d in s["disks"]):
            print(f"  disk layout (live): {s['disk_layout']}")
        print(f"  nics: {nics}")
        print()


def main_json(cfg, names=None, declared=None) -> str:
    si = connect(cfg)
    try:
        return json.dumps(summarize_lab(si, names, declared=declared), indent=2)
    finally:
        disconnect(si)
