"""Lab settings for testing preflight checks: disk.EnableUUID, RDM,
shared and independent disks, SR-IOV NICs. Offline tests — no vCenter
needed. Run from the project root:

    python -m pytest tests/ -q
    # or without pytest:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyVmomi import vim  # noqa: E402

from vcsim_lab import config, devices, query, specs, verify  # noqa: E402

LAB = os.path.join(os.path.dirname(__file__), "..", "environments", "default", "vms.yaml")
DEV_LAB = os.path.join(os.path.dirname(__file__), "..", "environments", "dev", "vms.yaml")

MINIMAL_VM = {
    "name": "x",
    "guest_id": "ubuntu64Guest",
    "firmware": "efi",
    "disks": [{"size_gb": 10}],
    "nics": [{"network": "prod"}],
}


def _load_vm(**overrides):
    raw = dict(MINIMAL_VM)
    raw.update(overrides)
    return config._load_vm(raw, [])


def _vm_spec(**overrides):
    base = dict(
        name="test-vm",
        guest_id="ubuntu64Guest",
        cpus=4,
        cores_per_socket=2,
        memory_mb=8192,
        hardware_version="vmx-20",
        firmware="efi",
        disks=[config.DiskSpec(size_gb=100, controller="nvme", datastore="LocalDS_0")],
        nics=[config.NicSpec(network="prod", type="vmxnet3")],
        boot_order=["disk"],
    )
    base.update(overrides)
    return config.VmSpec(**base)


def _refs():
    net = MagicMock(spec=vim.Network)
    net.name = "VM Network"
    return specs.ResolvedRefs(
        folder=MagicMock(), resource_pool=MagicMock(), datastores={},
        networks={"prod": net}, network_types={"prod": "standard"},
    )


def _extra(cfg):
    return {o.key: o.value for o in cfg.extraConfig or []}


def _mock_vm(extra_config=()):
    vm = MagicMock()
    vm.name = "test-vm"
    vm.config.files.vmPathName = "[LocalDS_0] test-vm/test-vm.vmx"
    vm.config.hardware.device = []
    vm.config.hardware.numCPU = 2
    vm.config.hardware.numCoresPerSocket = 1
    vm.config.hardware.memoryMB = 4096
    vm.config.guestId = "ubuntu64Guest"
    vm.config.guestFullName = "Ubuntu Linux (64-bit)"
    vm.config.annotation = ""
    vm.config.version = "vmx-20"
    vm.config.firmware = "bios"
    vm.config.template = False
    vm.config.bootOptions = None
    vm.config.vAppConfig = None
    vm.config.extraConfig = [
        SimpleNamespace(key=k, value=v) for k, v in extra_config
    ]
    vm.config.cpuHotAddEnabled = False
    vm.config.memoryHotAddEnabled = False
    vm.config.changeTrackingEnabled = True
    vm.datastore = []
    return vm


def _live(**overrides):
    base = {
        "name": "test-vm",
        "guest_id": "ubuntu64Guest",
        "cpus": 4,
        "memory_mb": 8192,
        "firmware": "efi",
        "hardware_version": "vmx-20",
        "power": "poweredOn",
        "boot_order": ["disk"],
        "template": False,
        "vapp": None,
        "disks": [
            {
                "capacity_gb": 100, "datastore": "LocalDS_0", "thin": True,
                "disk_mode": "persistent", "backing": "vmdk", "shared": False,
                "rdm_compatibility": None,
            }
        ],
        "nics": [{"label": "Network adapter 1", "type": "vim.vm.device.VirtualVmxnet3"}],
        "cdrom": False,
        "video_3d": False,
        "tpm": False,
        "secure_boot": False,
        "gpus": [],
        "cbt": True,
        "boot_delay_ms": 0,
        "tools_status": "toolsOk",
        "disk_enable_uuid": True,
    }
    base.update(overrides)
    return base


def _assigned(**overrides):
    base = dict(
        cbt=True, power_state="poweredOn", fast_boot=True,
        tools_status="toolsOk", disk_enable_uuid=True,
    )
    base.update(overrides)
    return config.ResolvedAssignment(**base)


class TestDiskEnableUuidConfig(unittest.TestCase):
    def test_bad_value_rejected(self):
        for bad in ("yes", 1, "sometimes"):
            with self.assertRaises(config.ConfigError):
                _load_vm(disk_enable_uuid=bad)

    def test_explicit_values_pass_through(self):
        lab = config.LabConfig(
            datacenter="DC0", cluster="DC0_C0", random_seed=42,
            vms=[
                _vm_spec(name="on", disk_enable_uuid=True),
                _vm_spec(name="off", disk_enable_uuid=False),
            ],
        )
        assigned = config.resolve_random_assignments(lab)
        self.assertIs(assigned["on"].disk_enable_uuid, True)
        self.assertIs(assigned["off"].disk_enable_uuid, False)

    def test_draw_last_does_not_move_earlier_draws(self):
        # The disk_enable_uuid draw happens after cbt / power_state /
        # fast_boot / tools_status, so resolving it must not change
        # any of the earlier draws for the same VMs + seed.
        vms_random = [_vm_spec(name=f"vm{i}") for i in range(5)]
        vms_fixed = [
            _vm_spec(name=f"vm{i}", disk_enable_uuid=False) for i in range(5)
        ]
        a_random = config.resolve_random_assignments(
            config.LabConfig(datacenter="DC0", cluster="DC0_C0", random_seed=42, vms=vms_random)
        )
        a_fixed = config.resolve_random_assignments(
            config.LabConfig(datacenter="DC0", cluster="DC0_C0", random_seed=42, vms=vms_fixed)
        )
        for name in a_random:
            for field in ("cbt", "power_state", "fast_boot", "tools_status"):
                self.assertEqual(
                    getattr(a_random[name], field), getattr(a_fixed[name], field), (name, field)
                )

    def test_resolved_assignment_defaults_false(self):
        assigned = config.ResolvedAssignment(
            cbt=True, power_state="poweredOn", fast_boot=True, tools_status="toolsOk"
        )
        self.assertIs(assigned.disk_enable_uuid, False)
        self.assertIn("disk_enable_uuid", assigned.to_dict())


class TestDiskSpecValidation(unittest.TestCase):
    def test_bad_disk_mode_rejected(self):
        with self.assertRaises(config.ConfigError):
            _load_vm(disks=[{"size_gb": 10, "disk_mode": "weird"}])

    def test_bad_backing_rejected(self):
        with self.assertRaises(config.ConfigError):
            _load_vm(disks=[{"size_gb": 10, "backing": "nfs"}])

    def test_bad_rdm_compatibility_rejected(self):
        with self.assertRaises(config.ConfigError):
            _load_vm(
                disks=[{"size_gb": 10, "backing": "rdm", "rdm_compatibility": "sideways"}]
            )

    def test_shared_rdm_rejected(self):
        with self.assertRaises(config.ConfigError):
            _load_vm(disks=[{"size_gb": 10, "backing": "rdm", "shared": True}])

    def test_valid_combinations_load(self):
        vm = _load_vm(
            disks=[
                {"size_gb": 10, "disk_mode": "independent-nonpersistent"},
                {"size_gb": 20, "backing": "rdm", "rdm_compatibility": "virtualCompatibility"},
                {"size_gb": 5, "shared": True},
            ]
        )
        self.assertEqual(vm.disks[0].disk_mode, "independent-nonpersistent")
        self.assertEqual(vm.disks[1].backing, "rdm")
        self.assertEqual(vm.disks[1].rdm_compatibility, "virtualCompatibility")
        self.assertTrue(vm.disks[2].shared)


class TestNicSpecValidation(unittest.TestCase):
    def test_sriov_with_type_rejected(self):
        with self.assertRaises(config.ConfigError):
            _load_vm(nics=[{"network": "prod", "type": "vmxnet3", "sriov": True}])

    def test_sriov_without_type_loads(self):
        vm = _load_vm(nics=[{"network": "prod", "sriov": True}])
        self.assertTrue(vm.nics[0].sriov)


class TestDeviceBuilders(unittest.TestCase):
    def test_rdm_device_name_deterministic(self):
        first = devices.rdm_device_name("vm1", 1)
        self.assertEqual(first, devices.rdm_device_name("vm1", 1))
        self.assertTrue(first.startswith("naa.6000c29"))
        self.assertNotEqual(first, devices.rdm_device_name("vm1", 2))
        self.assertNotEqual(first, devices.rdm_device_name("vm2", 1))

    def test_build_disk_rdm(self):
        disk = devices.build_disk(
            "vm1", 200, "LocalDS_2", -200, -100, 0, disk_index=2,
            backing="rdm", rdm_compatibility="physicalCompatibility",
        )
        backing = disk.device.backing
        self.assertIsInstance(
            backing, vim.vm.device.VirtualDisk.RawDiskMappingVer1BackingInfo
        )
        self.assertEqual(backing.compatibilityMode, "physicalCompatibility")
        self.assertEqual(backing.deviceName, devices.rdm_device_name("vm1", 2))
        self.assertEqual(backing.fileName, "[LocalDS_2] vm1/vm1_2.vmdk")
        self.assertFalse(hasattr(backing, "thinProvisioned"))

    def test_build_disk_shared(self):
        disk = devices.build_disk("vm1", 10, "LocalDS_0", -200, -100, 0, shared=True)
        self.assertEqual(disk.device.backing.sharing, "sharingMultiWriter")

    def test_build_disk_mode_passes_through(self):
        disk = devices.build_disk(
            "vm1", 100, "LocalDS_0", -200, -100, 0, disk_mode="independent-nonpersistent"
        )
        self.assertEqual(disk.device.backing.diskMode, "independent-nonpersistent")

    def test_build_nic_sriov(self):
        net = MagicMock(spec=vim.Network)
        net.name = "VM Network"
        nic = devices.build_nic(
            config.NicSpec(network="vm-network", sriov=True), -400, net, "standard"
        )
        self.assertIsInstance(nic.device, vim.vm.device.VirtualSriovEthernetCard)
        self.assertIsInstance(
            nic.device.backing,
            vim.vm.device.VirtualSriovEthernetCard.SriovBackingInfo,
        )

    def test_nic_device_class_name(self):
        self.assertEqual(
            devices.nic_device_class_name(config.NicSpec(network="p", sriov=True)),
            "vim.vm.device.VirtualSriovEthernetCard",
        )
        self.assertEqual(
            devices.nic_device_class_name(config.NicSpec(network="p", type="vmxnet3")),
            "vim.vm.device.VirtualVmxnet3",
        )
        self.assertEqual(
            devices.nic_device_class_name(config.NicSpec(network="p", type="e1000e")),
            "vim.vm.device.VirtualE1000e",
        )


class TestCreateSpec(unittest.TestCase):
    def test_disk_enable_uuid_stamped_true(self):
        cfg = specs.build_create_spec(
            _vm_spec(), specs.plan_vm(_vm_spec()), _refs(), disk_enable_uuid=True
        )
        self.assertEqual(_extra(cfg).get("disk.EnableUUID"), "TRUE")

    def test_disk_enable_uuid_stamped_false(self):
        cfg = specs.build_create_spec(
            _vm_spec(), specs.plan_vm(_vm_spec()), _refs(), disk_enable_uuid=False
        )
        self.assertEqual(_extra(cfg).get("disk.EnableUUID"), "FALSE")

    def test_disk_enable_uuid_defaults_off(self):
        cfg = specs.build_create_spec(_vm_spec(), specs.plan_vm(_vm_spec()), _refs())
        self.assertEqual(_extra(cfg).get("disk.EnableUUID"), "FALSE")

    def test_user_extra_config_wins(self):
        spec = _vm_spec(extra_config={"disk.EnableUUID": "TRUE"})
        cfg = specs.build_create_spec(
            spec, specs.plan_vm(spec), _refs(), disk_enable_uuid=False
        )
        self.assertEqual(_extra(cfg).get("disk.EnableUUID"), "TRUE")

    def test_rdm_and_shared_disks_in_spec(self):
        spec = _vm_spec(
            disks=[
                config.DiskSpec(size_gb=100, controller="pvscsi", datastore="LocalDS_0"),
                config.DiskSpec(
                    size_gb=200, controller="pvscsi", datastore="LocalDS_2",
                    backing="rdm", rdm_compatibility="virtualCompatibility",
                ),
                config.DiskSpec(size_gb=10, controller="pvscsi", datastore="LocalDS_0", shared=True),
            ]
        )
        cfg = specs.build_create_spec(spec, specs.plan_vm(spec), _refs())
        disks = [
            c.device for c in cfg.deviceChange
            if isinstance(c.device, vim.vm.device.VirtualDisk)
        ]
        self.assertEqual(len(disks), 3)
        self.assertIsInstance(
            disks[1].backing, vim.vm.device.VirtualDisk.RawDiskMappingVer1BackingInfo
        )
        self.assertEqual(disks[1].backing.compatibilityMode, "virtualCompatibility")
        self.assertEqual(disks[1].backing.deviceName, devices.rdm_device_name("test-vm", 1))
        self.assertEqual(disks[2].backing.sharing, "sharingMultiWriter")


class TestQuery(unittest.TestCase):
    def _disk_device(self, backing):
        disk = vim.vm.device.VirtualDisk()
        disk.deviceInfo = vim.Description(label="Hard disk 1", summary="100 GB")
        disk.capacityInKB = 100 * 1024 * 1024
        disk.controllerKey = -100
        disk.unitNumber = 0
        disk.backing = backing
        return disk

    def test_disk_info_rdm(self):
        backing = vim.vm.device.VirtualDisk.RawDiskMappingVer1BackingInfo()
        backing.deviceName = "naa.6000c29abc"
        backing.compatibilityMode = "physicalCompatibility"
        backing.fileName = "[LocalDS_2] vm1/vm1_2.vmdk"
        info = query._disk_info(self._disk_device(backing), {})
        self.assertEqual(info["backing"], "rdm")
        self.assertEqual(info["rdm_compatibility"], "physicalCompatibility")
        self.assertEqual(info["rdm_device_name"], "naa.6000c29abc")
        self.assertIsNone(info["thin"])
        self.assertIsNone(info["disk_mode"])
        self.assertFalse(info["shared"])

    def test_disk_info_shared(self):
        backing = vim.vm.device.VirtualDisk.FlatVer2BackingInfo()
        backing.diskMode = "persistent"
        backing.thinProvisioned = True
        backing.sharing = "sharingMultiWriter"
        backing.fileName = "[LocalDS_0] vm1/vm1.vmdk"
        info = query._disk_info(self._disk_device(backing), {})
        self.assertEqual(info["backing"], "vmdk")
        self.assertTrue(info["shared"])
        self.assertIsNone(info["rdm_compatibility"])

    def test_nic_info_sriov(self):
        card = vim.vm.device.VirtualSriovEthernetCard()
        card.deviceInfo = vim.Description(label="Network adapter 1", summary="SR-IOV")
        card.backing = vim.vm.device.VirtualSriovEthernetCard.SriovBackingInfo()
        card.macAddress = "00:0c:29:00:00:01"
        card.connectable = None
        info = query._nic_info(card)
        self.assertTrue(info["sriov"])
        self.assertEqual(info["type"], "vim.vm.device.VirtualSriovEthernetCard")

    def test_summarize_vm_disk_enable_uuid(self):
        summary = query.summarize_vm(
            _mock_vm([("disk.EnableUUID", "TRUE")])
        )
        self.assertIs(summary["disk_enable_uuid"], True)
        summary = query.summarize_vm(
            _mock_vm([("disk.EnableUUID", "FALSE")])
        )
        self.assertIs(summary["disk_enable_uuid"], False)
        summary = query.summarize_vm(_mock_vm())
        self.assertIsNone(summary["disk_enable_uuid"])


class TestVerify(unittest.TestCase):
    def test_matching_assignment_passes(self):
        failures, warnings = verify.verify_vm(_vm_spec(), _live(), _assigned())
        self.assertEqual(failures, [])
        self.assertEqual(warnings, [])

    def test_disk_enable_uuid_mismatch_is_a_failure(self):
        failures, _ = verify.verify_vm(
            _vm_spec(), _live(disk_enable_uuid=False), _assigned(disk_enable_uuid=True)
        )
        self.assertTrue(any("disk_enable_uuid" in f for f in failures))

    def test_disk_enable_uuid_skipped_when_live_unknown(self):
        failures, _ = verify.verify_vm(
            _vm_spec(), _live(disk_enable_uuid=None), _assigned(disk_enable_uuid=True)
        )
        self.assertEqual(failures, [])

    def test_rdm_disk_tuple_matches(self):
        spec = _vm_spec(
            disks=[
                config.DiskSpec(size_gb=100, controller="pvscsi", datastore="LocalDS_0"),
                config.DiskSpec(
                    size_gb=200, controller="lsi-sas", datastore="LocalDS_2", backing="rdm"
                ),
            ]
        )
        live = _live(
            disks=[
                {
                    "capacity_gb": 100, "datastore": "LocalDS_0", "thin": True,
                    "disk_mode": "persistent", "backing": "vmdk", "shared": False,
                    "rdm_compatibility": None,
                },
                {
                    "capacity_gb": 200, "datastore": "LocalDS_2", "thin": None,
                    "disk_mode": None, "backing": "rdm", "shared": False,
                    "rdm_compatibility": "physicalCompatibility",
                },
            ]
        )
        failures, _ = verify.verify_vm(spec, live, _assigned())
        self.assertEqual(failures, [])

    def test_disk_mode_mismatch_is_a_failure(self):
        spec = _vm_spec(
            disks=[
                config.DiskSpec(
                    size_gb=100, controller="nvme", datastore="LocalDS_0",
                    disk_mode="independent-nonpersistent",
                )
            ]
        )
        failures, _ = verify.verify_vm(spec, _live(), _assigned())
        self.assertTrue(any("disks" in f for f in failures))

    def test_shared_mismatch_is_a_failure(self):
        spec = _vm_spec(
            disks=[
                config.DiskSpec(size_gb=100, controller="nvme", datastore="LocalDS_0"),
                config.DiskSpec(size_gb=10, controller="nvme", datastore="LocalDS_0", shared=True),
            ]
        )
        live = _live(
            disks=[
                {
                    "capacity_gb": 100, "datastore": "LocalDS_0", "thin": True,
                    "disk_mode": "persistent", "backing": "vmdk", "shared": False,
                    "rdm_compatibility": None,
                },
                {
                    "capacity_gb": 10, "datastore": "LocalDS_0", "thin": True,
                    "disk_mode": "persistent", "backing": "vmdk", "shared": False,
                    "rdm_compatibility": None,
                },
            ]
        )
        failures, _ = verify.verify_vm(spec, live, _assigned())
        self.assertTrue(any("disks" in f for f in failures))

    def test_nic_types_match_including_sriov(self):
        spec = _vm_spec(
            nics=[
                config.NicSpec(network="prod", type="vmxnet3"),
                config.NicSpec(network="vm-network", sriov=True),
            ]
        )
        live = _live(
            nics=[
                {"label": "Network adapter 1", "type": "vim.vm.device.VirtualVmxnet3"},
                {"label": "Network adapter 2", "type": "vim.vm.device.VirtualSriovEthernetCard"},
            ]
        )
        failures, _ = verify.verify_vm(spec, live, _assigned())
        self.assertEqual(failures, [])

    def test_nic_type_mismatch_is_a_failure(self):
        spec = _vm_spec(nics=[config.NicSpec(network="prod", type="vmxnet3")])
        live = _live(nics=[{"label": "Network adapter 1", "type": "vim.vm.device.VirtualE1000"}])
        failures, _ = verify.verify_vm(spec, live, _assigned())
        self.assertTrue(any("nic types" in f for f in failures))

    def test_old_fixtures_without_new_keys_still_pass(self):
        # Live dicts from before these fields existed carry neither the
        # new disk keys nor NIC types — verify must not fail on them.
        live = _live(
            disks=[{"capacity_gb": 100, "datastore": "LocalDS_0", "thin": True}],
            nics=[{"label": "Network adapter 1"}],
        )
        del live["disk_enable_uuid"]
        failures, _ = verify.verify_vm(_vm_spec(), live, _assigned())
        self.assertEqual(failures, [])


class TestLabMatrix(unittest.TestCase):
    def test_default_lab_covers_new_settings(self):
        lab = config.load_lab(LAB)
        by_name = {v.name: v for v in lab.vms}
        # Explicit disk.EnableUUID on the two DB hosts; the rest resolve
        # from the seed and cover both values (pinned in test_random_attrs).
        self.assertIs(by_name["win2022-sql"].disk_enable_uuid, True)
        self.assertIs(by_name["rhel9-db"].disk_enable_uuid, True)
        # One RDM disk, one shared disk, one independent-nonpersistent
        # disk, one SR-IOV NIC — each on the VM the matrix assigns it to.
        rdm = [
            (v.name, d) for v in lab.vms for d in v.disks if d.backing == "rdm"
        ]
        self.assertEqual([n for n, _ in rdm], ["freebsd13-nas"])
        self.assertEqual(rdm[0][1].rdm_compatibility, "physicalCompatibility")
        shared = [
            (v.name, d) for v in lab.vms for d in v.disks if d.shared
        ]
        self.assertEqual([n for n, _ in shared], ["sles15-sap"])
        nonpersistent = [
            (v.name, d)
            for v in lab.vms
            for d in v.disks
            if d.disk_mode == "independent-nonpersistent"
        ]
        self.assertEqual([n for n, _ in nonpersistent], ["centos8-legacy"])
        sriov = [
            (v.name, n) for v in lab.vms for n in v.nics if n.sriov
        ]
        self.assertEqual([n for n, _ in sriov], ["opnsense-fw"])
        # 52 disks (was 46 — the big HANA VM adds three 1 TB disks,
        # the other three new VMs one disk each).
        self.assertEqual(sum(len(v.disks) for v in lab.vms), 52)

    def test_dev_lab_loads_with_enable_uuid(self):
        lab = config.load_lab(DEV_LAB)
        by_name = {v.name: v for v in lab.vms}
        self.assertIs(by_name["dev-win10"].disk_enable_uuid, True)
