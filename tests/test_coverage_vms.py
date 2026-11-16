"""Coverage VMs for the five preflight checks no lab VM exercised
(entry 22): cpu-memory-size, disk-size, nested-virt,
unsupported-devices, vm-name. Plus the two builder fields they need:
`nested_virt` (create spec) and `extra_devices` (post-create reconfig).
Offline tests — no vCenter needed. Run from the project root:

    python -m pytest tests/ -q
"""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyVmomi import vim  # noqa: E402

from vcsim_lab import build, config, devices, query, specs, verify  # noqa: E402

LAB = os.path.join(os.path.dirname(__file__), "..", "environments", "default", "vms.yaml")

MINIMAL_VM = {
    "name": "x",
    "guest_id": "ubuntu64Guest",
    "firmware": "efi",
    "disks": [{"size_gb": 10}],
    "nics": [{"network": "prod"}],
}

BIG_NAME = "sap-hana-production-database-server-primary-001-eu-central-instance"


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


def _mock_vm(extra_config=(), nested_virt=False, hw_devices=()):
    vm = MagicMock()
    vm.name = "test-vm"
    vm.config.files.vmPathName = "[LocalDS_0] test-vm/test-vm.vmx"
    vm.config.hardware.device = list(hw_devices)
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
    vm.config.nestedHVEnabled = nested_virt
    vm.config.changeTrackingEnabled = True
    vm.datastore = []
    return vm


class TestConfigFields(unittest.TestCase):
    def test_nested_virt_and_extra_devices_load(self):
        vm = _load_vm(nested_virt=True, extra_devices=["usb", "floppy"])
        self.assertTrue(vm.nested_virt)
        self.assertEqual(vm.extra_devices, ["usb", "floppy"])

    def test_defaults_are_off_and_empty(self):
        vm = _load_vm()
        self.assertFalse(vm.nested_virt)
        self.assertEqual(vm.extra_devices, [])

    def test_unknown_extra_device_rejected(self):
        with self.assertRaises(config.ConfigError):
            _load_vm(extra_devices=["usb", "thunderbolt"])

    def test_duplicate_extra_device_rejected(self):
        with self.assertRaises(config.ConfigError):
            _load_vm(extra_devices=["usb", "usb"])


class TestCreateSpecNestedVirt(unittest.TestCase):
    def test_nested_hv_enabled_in_create_spec(self):
        spec = _vm_spec(nested_virt=True)
        cfg = specs.build_create_spec(spec, specs.plan_vm(spec), _refs())
        self.assertTrue(cfg.nestedHVEnabled)

    def test_nested_hv_disabled_by_default(self):
        spec = _vm_spec()
        cfg = specs.build_create_spec(spec, specs.plan_vm(spec), _refs())
        self.assertFalse(cfg.nestedHVEnabled)


class TestExtraDeviceBuilders(unittest.TestCase):
    def test_each_kind_builds_its_class(self):
        expected = {
            "usb": vim.vm.device.VirtualUSBController,
            "serial": vim.vm.device.VirtualSerialPort,
            "parallel": vim.vm.device.VirtualParallelPort,
            "sound": vim.vm.device.VirtualSoundCard,
            "floppy": vim.vm.device.VirtualFloppy,
        }
        keys = set()
        for kind, cls in expected.items():
            device_spec = devices.build_extra_device(kind)
            self.assertIsInstance(device_spec.device, cls)
            self.assertLess(device_spec.device.key, 0)  # negative temp key
            keys.add(device_spec.device.key)
        self.assertEqual(len(keys), 5)  # distinct keys per kind

    def test_extra_device_kinds_detects_present(self):
        hw = [
            vim.vm.device.VirtualUSBController(),
            vim.vm.device.VirtualFloppy(),
            vim.vm.device.VirtualCdrom(),
        ]
        self.assertEqual(devices.extra_device_kinds(hw), ["usb", "floppy"])

    def test_extra_device_kinds_empty(self):
        self.assertEqual(devices.extra_device_kinds([]), [])


class TestAddExtraDevices(unittest.TestCase):
    def _mock_vm_with(self, hw_devices):
        vm = MagicMock()
        vm.config.hardware.device = list(hw_devices)
        return vm

    def test_adds_missing_devices_in_one_reconfig(self):
        vm = self._mock_vm_with([vim.vm.device.VirtualUSBController()])
        spec = _vm_spec(extra_devices=["usb", "serial", "floppy"])
        report = build.VmReport(name="test-vm", created=True)
        with patch.object(build, "wait_for_task"):
            build.add_extra_devices(MagicMock(), vm, spec, report)
        vm.ReconfigVM_Task.assert_called_once()
        cfg = vm.ReconfigVM_Task.call_args.args[0]
        added = {type(d.device).__name__ for d in cfg.deviceChange}
        self.assertEqual(
            added,
            {"vim.vm.device.VirtualSerialPort", "vim.vm.device.VirtualFloppy"},
        )
        self.assertEqual(report.steps[-1].status, "ok")

    def test_skips_when_all_present(self):
        vm = self._mock_vm_with([
            vim.vm.device.VirtualUSBController(),
            vim.vm.device.VirtualSerialPort(),
        ])
        spec = _vm_spec(extra_devices=["usb", "serial"])
        report = build.VmReport(name="test-vm", created=True)
        build.add_extra_devices(MagicMock(), vm, spec, report)
        vm.ReconfigVM_Task.assert_not_called()
        self.assertEqual(report.steps[-1].status, "skipped")


class TestQuerySummary(unittest.TestCase):
    def test_summarize_vm_reports_nested_virt_and_extra_devices(self):
        vm = _mock_vm(
            nested_virt=True,
            hw_devices=[vim.vm.device.VirtualUSBController(), vim.vm.device.VirtualSoundCard()],
        )
        summary = query.summarize_vm(vm)
        self.assertTrue(summary["nested_virt"])
        self.assertEqual(summary["extra_devices"], ["usb", "sound"])

    def test_summarize_vm_defaults(self):
        summary = query.summarize_vm(_mock_vm())
        self.assertFalse(summary["nested_virt"])
        self.assertEqual(summary["extra_devices"], [])


class TestVerifyCoverage(unittest.TestCase):
    def test_nested_virt_mismatch_is_a_failure(self):
        spec = _vm_spec(nested_virt=True)
        failures, _ = verify.verify_vm(spec, _live(nested_virt=False), _assigned())
        self.assertTrue(any("nested_virt" in f for f in failures))
        failures, _ = verify.verify_vm(spec, _live(nested_virt=True), _assigned())
        self.assertEqual(failures, [])

    def test_extra_devices_mismatch_is_a_failure(self):
        spec = _vm_spec(extra_devices=["usb", "floppy"])
        failures, _ = verify.verify_vm(
            spec, _live(extra_devices=["usb"]), _assigned()
        )
        self.assertTrue(any("extra_devices" in f for f in failures))
        failures, _ = verify.verify_vm(
            spec, _live(extra_devices=["floppy", "usb"]), _assigned()
        )
        self.assertEqual(failures, [])

    def test_missing_live_values_skip(self):
        # Older summaries predate the fields — no failure, no warning.
        spec = _vm_spec(nested_virt=True, extra_devices=["usb"])
        live = _live()
        live.pop("nested_virt", None)
        live.pop("extra_devices", None)
        failures, warnings = verify.verify_vm(spec, live, _assigned())
        self.assertEqual(failures, [])


class TestLabMatrix(unittest.TestCase):
    def test_the_four_coverage_vms(self):
        lab = config.load_lab(LAB)
        by_name = {v.name: v for v in lab.vms}
        # The oversized HANA server: over every preflight size
        # threshold, and its name is over the 63-char K8s limit.
        big = by_name[BIG_NAME]
        self.assertEqual(len(BIG_NAME), 67)
        self.assertEqual(big.cpus, 48)
        self.assertEqual(big.memory_mb, 524288)
        self.assertEqual(sum(d.size_gb for d in big.disks), 3072)
        self.assertEqual(big.folder, "lab/Prod/DB")
        # The invalid name: uppercase + underscore, not DNS-1123.
        bad = by_name["Win2019_FileServer"]
        self.assertEqual(bad.folder, "lab/Lab/Legacy")
        # The nested host.
        nested = by_name["kvm-nested-host"]
        self.assertTrue(nested.nested_virt)
        self.assertEqual(nested.folder, "lab/Lab")
        # The legacy desktop carries all five unsupported devices.
        desk = by_name["win10-legacy-desk"]
        self.assertEqual(
            sorted(desk.extra_devices),
            ["floppy", "parallel", "serial", "sound", "usb"],
        )
        self.assertEqual(desk.folder, "lab/Prod/VDI")

    def test_lab_now_exceeds_preflight_size_thresholds(self):
        # cpu-memory-size warns over 32 vCPUs / 256 GB; disk-size
        # warns over 2048 GB total. Before entry 22 the lab maxed
        # out at 16 vCPUs / 64 GB / 1100 GB.
        lab = config.load_lab(LAB)
        self.assertTrue(any(v.cpus > 32 for v in lab.vms))
        self.assertTrue(any(v.memory_mb > 256 * 1024 for v in lab.vms))
        self.assertTrue(
            any(sum(d.size_gb for d in v.disks) > 2048 for v in lab.vms)
        )
