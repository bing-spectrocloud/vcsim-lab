"""Offline tests for VM snapshots, the Solaris guest OS and the
explicit toolsOk spread: no vCenter needed.

Run from the project root:

    python -m pytest tests/ -q
    # or without pytest:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import datetime
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vcsim_lab import build, config, query, verify  # noqa: E402

LAB = os.path.join(os.path.dirname(__file__), "..", "environments", "default", "vms.yaml")

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


class TestSnapshotConfig(unittest.TestCase):
    def test_snapshots_parse_with_description(self):
        vm = _load_vm(
            snapshots=[
                {"name": "baseline", "description": "Fresh install"},
                {"name": "pre-patch"},
            ]
        )
        self.assertEqual(
            [(s.name, s.description) for s in vm.snapshots],
            [("baseline", "Fresh install"), ("pre-patch", "")],
        )

    def test_snapshots_default_to_empty(self):
        self.assertEqual(_load_vm().snapshots, [])

    def test_duplicate_snapshot_name_rejected(self):
        with self.assertRaises(config.ConfigError):
            _load_vm(snapshots=[{"name": "baseline"}, {"name": "baseline"}])

    def test_snapshot_without_name_rejected(self):
        with self.assertRaises(config.ConfigError):
            _load_vm(snapshots=[{"description": "no name"}])
        with self.assertRaises(config.ConfigError):
            _load_vm(snapshots=[{"name": ""}])

    def test_non_mapping_snapshot_rejected(self):
        with self.assertRaises(config.ConfigError):
            _load_vm(snapshots=["baseline"])

    def test_template_with_snapshots_rejected(self):
        with self.assertRaises(config.ConfigError):
            _load_vm(template=True, snapshots=[{"name": "baseline"}])


class TestSolarisGuestId(unittest.TestCase):
    def test_solaris_ids_are_known(self):
        for guest_id in ("solaris11_64Guest", "solaris11Guest",
                         "solaris10_64Guest", "solaris10Guest"):
            self.assertIn(guest_id, config.KNOWN_GUEST_IDS)

    def test_solaris_vm_loads_without_warning(self):
        warnings = []
        vm = config._load_vm({**MINIMAL_VM, "guest_id": "solaris11_64Guest"}, warnings)
        self.assertEqual(vm.guest_id, "solaris11_64Guest")
        self.assertEqual(warnings, [])

    def test_unknown_guest_id_still_warns(self):
        warnings = []
        config._load_vm({**MINIMAL_VM, "guest_id": "notAnOsGuest"}, warnings)
        self.assertEqual(len(warnings), 1)


class TestLabMatrix(unittest.TestCase):
    def test_lab_has_26_vms(self):
        self.assertEqual(len(config.load_lab(LAB).vms), 26)

    def test_solaris_vm(self):
        vm = config.load_lab(LAB).vm("solaris11-db")
        self.assertEqual(vm.guest_id, "solaris11_64Guest")
        self.assertEqual(vm.os_label, "Oracle Solaris 11 (64-bit)")
        self.assertEqual(vm.firmware, "bios")
        self.assertFalse(vm.tpm)
        self.assertFalse(vm.secure_boot)
        self.assertEqual(vm.cpus, 2)
        self.assertEqual(vm.memory_mb, 4096)
        self.assertEqual(vm.folder, "lab/Prod/DB")
        self.assertEqual([n.network for n in vm.nics], ["vm-network"])
        self.assertEqual(len(vm.disks), 2)
        self.assertEqual(vm.tools_status, "toolsOk")
        self.assertEqual([s.name for s in vm.snapshots], ["baseline"])

    def test_snapshot_distribution_covers_0_1_2(self):
        lab = config.load_lab(LAB)
        counts = {}
        for vm in lab.vms:
            counts[len(vm.snapshots)] = counts.get(len(vm.snapshots), 0) + 1
        self.assertEqual(counts, {0: 15, 1: 6, 2: 5})
        self.assertEqual(sum(len(v.snapshots) for v in lab.vms), 16)

    def test_no_template_has_snapshots(self):
        lab = config.load_lab(LAB)
        for vm in lab.vms:
            if vm.template:
                self.assertEqual(vm.snapshots, [], vm.name)

    def test_explicit_tools_ok_spans_os_families(self):
        lab = config.load_lab(LAB)
        ok = [v for v in lab.vms if v.tools_status == "toolsOk"]
        self.assertGreaterEqual(len(ok), 4)
        # Windows, Debian/Ubuntu Linux, Solaris and FreeBSD all covered.
        families = set()
        for vm in ok:
            if vm.guest_id.startswith("windows"):
                families.add("windows")
            elif "solaris" in vm.guest_id:
                families.add("solaris")
            elif "freebsd" in vm.guest_id:
                families.add("freebsd")
            else:
                families.add("linux")
        self.assertEqual(families, {"windows", "linux", "solaris", "freebsd"})
        # And the seed-resolved assignment still declares all four
        # tools statuses somewhere in the lab.
        assigned = config.resolve_random_assignments(lab)
        self.assertEqual(
            {a.tools_status for a in assigned.values()}, set(config.TOOLS_STATUSES)
        )

    def test_declared_assignments_include_snapshots(self):
        declared = config.load_declared_assignments(LAB)
        self.assertEqual(
            declared["win2022-sql"]["snapshots"], ["baseline", "pre-patch"]
        )
        self.assertEqual(declared["solaris11-db"]["snapshots"], ["baseline"])
        self.assertEqual(declared["centos8-legacy"]["snapshots"], [])


class TestBuildSnapshots(unittest.TestCase):
    def test_apply_snapshots_creates_each_in_order(self):
        spec = _vm_spec(
            snapshots=[
                config.SnapshotSpec(name="baseline", description="Fresh"),
                config.SnapshotSpec(name="pre-patch"),
            ]
        )
        vm = MagicMock()
        task = MagicMock()
        vm.CreateSnapshot.return_value = task
        report = build.VmReport(name="test-vm", created=True)
        with patch("vcsim_lab.build.wait_for_task") as wait:
            build.apply_snapshots(MagicMock(), vm, spec, report)
        self.assertEqual(vm.CreateSnapshot.call_count, 2)
        for call, name, description in zip(
            vm.CreateSnapshot.call_args_list,
            ["baseline", "pre-patch"],
            ["Fresh", ""],
        ):
            kwargs = call.kwargs
            self.assertEqual(kwargs["name"], name)
            self.assertEqual(kwargs["description"], description)
            # No memory dump, no quiesce: snapshots work in any power
            # state without guest cooperation.
            self.assertIs(kwargs["memory"], False)
            self.assertIs(kwargs["quiesce"], False)
        self.assertEqual(wait.call_count, 2)
        self.assertEqual([s.step for s in report.steps], ["snapshots"])
        self.assertEqual(report.steps[0].status, "ok")
        self.assertIn("baseline", report.steps[0].detail)

    def test_apply_snapshots_skips_when_none_declared(self):
        spec = _vm_spec()
        vm = MagicMock()
        report = build.VmReport(name="test-vm", created=True)
        with patch("vcsim_lab.build.wait_for_task") as wait:
            build.apply_snapshots(MagicMock(), vm, spec, report)
        vm.CreateSnapshot.assert_not_called()
        wait.assert_not_called()
        self.assertEqual(report.steps[0].status, "skipped")


def _tree(name, moid, children=None, state="poweredOff", description=""):
    node = MagicMock()
    node.name = name
    node.description = description
    node.id = hash(name) % 1000
    node.createTime = datetime.datetime(2026, 11, 14, 12, 0, 0)
    node.state = state
    node.snapshot = MagicMock()
    node.snapshot._moId = moid
    node.childSnapshotList = children or []
    return node


class TestQuerySnapshots(unittest.TestCase):
    def test_flatten_tree_is_preorder_and_maps_moids(self):
        child = _tree("child", "snap-2")
        root = _tree("root", "snap-1", children=[child])
        entries, by_moid = query._snapshot_tree_entries([root])
        self.assertEqual([e["name"] for e in entries], ["root", "child"])
        self.assertEqual(entries[0]["state"], "poweredOff")
        self.assertEqual(entries[0]["create_time"], "2026-11-14T12:00:00")
        self.assertEqual(by_moid, {"snap-1": "root", "snap-2": "child"})

    def test_flatten_empty_tree(self):
        self.assertEqual(query._snapshot_tree_entries(None), ([], {}))
        self.assertEqual(query._snapshot_tree_entries([]), ([], {}))

    def test_retrieve_snapshot_props_resolves_current(self):
        vm = MagicMock()
        vm._moId = "vm-1"
        root = _tree("baseline", "snap-1")
        current = MagicMock()
        current._moId = "snap-1"
        leaf = {
            "vm-1": {
                "snapshot.rootSnapshotList": [root],
                "snapshot.currentSnapshot": current,
            }
        }
        with patch("vcsim_lab.query.retrieve_props_multi", return_value=leaf):
            props = query._retrieve_snapshot_props(MagicMock(), [vm])
        self.assertEqual([s["name"] for s in props["vm-1"]["snapshots"]], ["baseline"])
        self.assertEqual(props["vm-1"]["current_snapshot"], "baseline")

    def test_retrieve_snapshot_props_unknown_when_unreadable(self):
        vm = MagicMock()
        vm._moId = "vm-1"
        with (
            patch("vcsim_lab.query.retrieve_props_multi", side_effect=RuntimeError("boom")),
            patch("vcsim_lab.query.retrieve_props", side_effect=RuntimeError("boom")),
        ):
            props = query._retrieve_snapshot_props(MagicMock(), [vm])
        # Unknown, not zero: verify must skip the check, not fail it.
        self.assertIsNone(props["vm-1"]["snapshots"])
        self.assertIsNone(props["vm-1"]["current_snapshot"])

    def test_retrieve_snapshot_props_empty_when_no_snapshots(self):
        vm = MagicMock()
        vm._moId = "vm-1"
        with patch("vcsim_lab.query.retrieve_props_multi", return_value={"vm-1": {}}):
            props = query._retrieve_snapshot_props(MagicMock(), [vm])
        self.assertEqual(props["vm-1"]["snapshots"], [])
        self.assertIsNone(props["vm-1"]["current_snapshot"])

    def _mock_vm(self):
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
        vm.config.extraConfig = []
        vm.config.cpuHotAddEnabled = False
        vm.config.memoryHotAddEnabled = False
        vm.config.changeTrackingEnabled = True
        vm.datastore = []
        return vm

    def test_summarize_vm_carries_snapshots(self):
        vm = self._mock_vm()
        snapshots = [
            {"name": "baseline", "description": "", "id": 1,
             "state": "poweredOff", "create_time": "2026-11-14T12:00:00"},
            {"name": "pre-patch", "description": "", "id": 2,
             "state": "poweredOff", "create_time": "2026-11-14T12:01:00"},
        ]
        summary = query.summarize_vm(
            vm,
            runtime={
                "power": "poweredOff",
                "cbt": True,
                "boot_delay_ms": 3000,
                "tools_status": "toolsNotInstalled",
                "snapshots": snapshots,
                "current_snapshot": "pre-patch",
            },
            declared={"tools_status": "toolsOk", "snapshots": ["baseline", "pre-patch"]},
        )
        self.assertEqual(summary["snapshots"], snapshots)
        self.assertEqual(summary["snapshot_count"], 2)
        self.assertEqual(summary["current_snapshot"], "pre-patch")
        self.assertIs(summary["snapshot_mismatch"], False)

    def test_summarize_vm_flags_snapshot_mismatch(self):
        vm = self._mock_vm()
        summary = query.summarize_vm(
            vm,
            runtime={
                "power": "poweredOff",
                "cbt": True,
                "boot_delay_ms": 3000,
                "tools_status": "toolsNotInstalled",
                "snapshots": [],
                "current_snapshot": None,
            },
            declared={"tools_status": "toolsOk", "snapshots": ["baseline"]},
        )
        self.assertEqual(summary["snapshot_count"], 0)
        self.assertIs(summary["snapshot_mismatch"], True)

    def test_summarize_vm_unknown_snapshots_stays_none(self):
        vm = self._mock_vm()
        summary = query.summarize_vm(
            vm,
            runtime={
                "power": "poweredOff",
                "cbt": True,
                "boot_delay_ms": 3000,
                "tools_status": "toolsNotInstalled",
                "snapshots": None,
                "current_snapshot": None,
            },
            declared={"tools_status": "toolsOk", "snapshots": ["baseline"]},
        )
        self.assertIsNone(summary["snapshots"])
        self.assertIsNone(summary["snapshot_count"])
        # No mismatch flag when the live state is unknown.
        self.assertNotIn("snapshot_mismatch", summary)


def _live(**overrides):
    base = {
        "name": "test-vm",
        "guest_id": "ubuntu64Guest",
        "cpus": 4,
        "memory_mb": 8192,
        "firmware": "efi",
        "hardware_version": "vmx-20",
        "power": "poweredOff",
        "boot_order": ["disk"],
        "template": False,
        "vapp": None,
        "disks": [{"capacity_gb": 100, "datastore": "LocalDS_0", "thin": True}],
        "nics": [{"label": "Network adapter 1"}],
        "cdrom": False,
        "video_3d": False,
        "tpm": False,
        "secure_boot": False,
        "gpus": [],
        "cbt": False,
        "boot_delay_ms": 3000,
        "tools_status": "toolsNotInstalled",
        "snapshots": [],
    }
    base.update(overrides)
    return base


class TestVerifySnapshots(unittest.TestCase):
    def test_matching_snapshots_pass(self):
        spec = _vm_spec(
            snapshots=[config.SnapshotSpec(name="baseline"),
                       config.SnapshotSpec(name="pre-patch")]
        )
        live = _live(snapshots=[{"name": "baseline"}, {"name": "pre-patch"}])
        failures, warnings = verify.verify_vm(spec, live)
        self.assertEqual(failures, [])
        self.assertEqual(warnings, [])

    def test_missing_snapshot_fails(self):
        spec = _vm_spec(snapshots=[config.SnapshotSpec(name="baseline")])
        failures, _ = verify.verify_vm(spec, _live(snapshots=[]))
        self.assertTrue(any("snapshots" in f for f in failures))

    def test_extra_snapshot_fails(self):
        spec = _vm_spec()
        failures, _ = verify.verify_vm(spec, _live(snapshots=[{"name": "rogue"}]))
        self.assertTrue(any("snapshots" in f for f in failures))

    def test_wrong_snapshot_name_fails(self):
        spec = _vm_spec(snapshots=[config.SnapshotSpec(name="baseline")])
        failures, _ = verify.verify_vm(spec, _live(snapshots=[{"name": "other"}]))
        self.assertTrue(any("snapshots" in f for f in failures))

    def test_unknown_snapshot_state_skips_check(self):
        spec = _vm_spec(snapshots=[config.SnapshotSpec(name="baseline")])
        failures, warnings = verify.verify_vm(spec, _live(snapshots=None))
        self.assertEqual(failures, [])
        self.assertFalse(any("snapshots" in w for w in warnings))

    def test_no_snapshots_declared_or_live_passes(self):
        failures, warnings = verify.verify_vm(_vm_spec(), _live())
        self.assertEqual(failures, [])
        self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
