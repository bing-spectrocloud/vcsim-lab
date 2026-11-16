"""Offline tests for nested VM folders: no vCenter needed.

Covers folder-path validation, the default lab's nested hierarchy,
recursive folder creation, the parent-chain path walk, the verify
folder check and the folder field in VM summaries.

Run from the project root:

    python -m pytest tests/ -q
    # or without pytest:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

from pyVmomi import vim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vcsim_lab import build, config, query, vcenter, verify  # noqa: E402

LAB = os.path.join(os.path.dirname(__file__), "..", "environments", "default", "vms.yaml")

MINIMAL_VM = {
    "name": "x",
    "guest_id": "ubuntu64Guest",
    "firmware": "efi",
    "disks": [{"size_gb": 10}],
    "nics": [{"network": "prod"}],
}

EXPECTED_FOLDERS = [
    "lab",
    "lab/Prod",
    "lab/Prod/Web",
    "lab/Prod/DB",
    "lab/Prod/VDI",
    "lab/Lab",
    "lab/Lab/GPU",
    "lab/Lab/Legacy",
    "lab/Templates",
    "lab/Appliances",
]

# VM name -> folder path, the default lab's hierarchy.
EXPECTED_VM_FOLDERS = {
    "win11-secure": "lab/Prod",
    "win2022-sql": "lab/Prod/DB",
    "win10-vdi": "lab/Prod/VDI",
    "win2019-fileserver": "lab/Prod",
    "win2008r2-legacy": "lab/Lab/Legacy",
    "winxp64-legacy": "lab/Lab/Legacy",
    "Win2019_FileServer": "lab/Lab/Legacy",
    "win10-legacy-desk": "lab/Prod/VDI",
    "ubuntu-gpu-llm": "lab/Lab/GPU",
    "ubuntu-vgpu-vdi": "lab/Prod/VDI",
    "rhel9-ai-train": "lab/Lab/GPU",
    "debian12-web": "lab/Prod/Web",
    "photon-container": "lab/Prod/Web",
    "rhel9-db": "lab/Prod/DB",
    "sles15-sap": "lab/Prod/Web",
    "sap-hana-production-database-server-primary-001-eu-central-instance": "lab/Prod/DB",
    "freebsd13-nas": "lab",
    "solaris11-db": "lab/Prod/DB",
    "centos8-legacy": "lab/Lab/Legacy",
    "kvm-nested-host": "lab/Lab",
    "ubuntu-k8s-node": "lab/Prod",
    "ubuntu-nvidia-a100-pt": "lab/Lab/GPU",
    "ubuntu-nvidia-h100-vgpu": "lab/Lab/GPU",
    "rhel9-amd-mi210-pt": "lab/Lab/GPU",
    "win11-amd-mxgpu": "lab/Prod/VDI",
    "win2022-template": "lab/Templates",
    "ubuntu-2404-template": "lab/Templates",
    "rhel9-template": "lab/Templates",
    "opnsense-fw": "lab/Appliances",
    "truenas-storage": "lab/Appliances",
}


def _load_vm(**overrides):
    raw = dict(MINIMAL_VM)
    raw.update(overrides)
    return config._load_vm(raw, [])


def _folder_mock(name, parent=None):
    """A Folder mock whose CreateFolder really appends to childEntity,
    so recursive creation can be observed and re-runs find the folders
    the first run made."""
    folder = MagicMock(spec=vim.Folder)
    folder.name = name
    folder._moId = f"folder-{name}-{id(folder)}"
    folder.childEntity = []
    folder.parent = parent

    def _create(child_name):
        child = _folder_mock(child_name, parent=folder)
        folder.childEntity.append(child)
        return child

    folder.CreateFolder.side_effect = _create
    return folder


def _dc_mock():
    dc = MagicMock(spec=vim.Datacenter)
    dc.name = "DC0"
    vm_folder = _folder_mock("vm", parent=dc)
    dc.vmFolder = vm_folder
    return dc, vm_folder


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
        "folder": "lab/Prod/Web",
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


def _vm_spec(**overrides):
    base = dict(
        name="test-vm",
        guest_id="ubuntu64Guest",
        cpus=4,
        cores_per_socket=2,
        memory_mb=8192,
        hardware_version="vmx-20",
        firmware="efi",
        folder="lab/Prod/Web",
        disks=[config.DiskSpec(size_gb=100, controller="nvme", datastore="LocalDS_0")],
        nics=[config.NicSpec(network="prod", type="vmxnet3")],
        boot_order=["disk"],
    )
    base.update(overrides)
    return config.VmSpec(**base)


def _mock_vm():
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


class TestValidateFolderPath(unittest.TestCase):
    def test_valid_paths_pass_through(self):
        for path in ("", "lab", "lab/Prod", "lab/Prod/Web"):
            self.assertEqual(config.validate_folder_path(path, "test:"), path)

    def test_leading_or_trailing_slash_rejected(self):
        for path in ("/lab", "lab/", "/lab/Prod/", "/"):
            with self.assertRaises(config.ConfigError):
                config.validate_folder_path(path, "test:")

    def test_empty_segment_rejected(self):
        for path in ("lab//Web", "lab///Prod"):
            with self.assertRaises(config.ConfigError):
                config.validate_folder_path(path, "test:")

    def test_dot_segments_rejected(self):
        for path in ("lab/./Web", "lab/../Web", ".", "..", "lab/Prod/.."):
            with self.assertRaises(config.ConfigError):
                config.validate_folder_path(path, "test:")

    def test_non_string_rejected(self):
        with self.assertRaises(config.ConfigError):
            config.validate_folder_path(None, "test:")
        with self.assertRaises(config.ConfigError):
            config.validate_folder_path(42, "test:")

    def test_vm_with_bad_folder_rejected(self):
        with self.assertRaises(config.ConfigError):
            _load_vm(folder="/lab/Prod")
        with self.assertRaises(config.ConfigError):
            _load_vm(folder="lab//Prod")

    def test_vm_folder_defaults_to_root(self):
        self.assertEqual(_load_vm().folder, "")


class TestLoadLabFolders(unittest.TestCase):
    def _write_lab(self, folders_yaml: str) -> str:
        text = f"""lab:
  datacenter: DC0
  cluster: DC0_C0
  folders:
{folders_yaml}
networks:
  prod:
    name: DC0_DVPG0
    type: dvpg
vms:
  - name: x
    guest_id: ubuntu64Guest
    disks: [{{size_gb: 10}}]
    nics: [{{network: prod}}]
"""
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        tmp.write(text)
        tmp.close()
        return tmp.name

    def test_bad_lab_folder_rejected(self):
        path = self._write_lab("    - lab/../Prod\n")
        try:
            with self.assertRaises(config.ConfigError):
                config.load_lab(path)
        finally:
            os.unlink(path)

    def test_duplicate_lab_folder_rejected(self):
        path = self._write_lab("    - lab/Prod\n    - lab/Prod\n")
        try:
            with self.assertRaises(config.ConfigError):
                config.load_lab(path)
        finally:
            os.unlink(path)

    def test_good_lab_folders_load(self):
        path = self._write_lab("    - lab\n    - lab/Prod/Web\n")
        try:
            lab = config.load_lab(path)
        finally:
            os.unlink(path)
        self.assertEqual(lab.folders, ["lab", "lab/Prod/Web"])


class TestLabFolderMatrix(unittest.TestCase):
    def test_folders_list(self):
        self.assertEqual(config.load_lab(LAB).folders, EXPECTED_FOLDERS)

    def test_vm_folder_assignments(self):
        lab = config.load_lab(LAB)
        actual = {vm.name: vm.folder for vm in lab.vms}
        self.assertEqual(actual, EXPECTED_VM_FOLDERS)

    def test_every_vm_folder_is_listed(self):
        lab = config.load_lab(LAB)
        listed = set(lab.folders)
        for vm in lab.vms:
            self.assertIn(vm.folder, listed, f"{vm.name} folder not in lab.folders")

    def test_distribution_per_folder(self):
        lab = config.load_lab(LAB)
        counts: dict[str, int] = {}
        for vm in lab.vms:
            counts[vm.folder] = counts.get(vm.folder, 0) + 1
        self.assertEqual(
            counts,
            {
                "lab": 1,
                "lab/Prod": 3,
                "lab/Prod/Web": 3,
                "lab/Prod/DB": 4,
                "lab/Prod/VDI": 4,
                "lab/Lab": 1,
                "lab/Lab/GPU": 5,
                "lab/Lab/Legacy": 4,
                "lab/Templates": 3,
                "lab/Appliances": 2,
            },
        )

    def test_three_levels_deep_and_vms_at_intermediate_levels(self):
        lab = config.load_lab(LAB)
        depths = {len(vm.folder.split("/")) for vm in lab.vms}
        self.assertIn(3, depths)  # e.g. lab/Prod/Web
        self.assertIn(2, depths)  # e.g. lab/Prod
        self.assertIn(1, depths)  # directly under lab
        # Intermediate folders hold VMs directly, not only leaves.
        by_folder: dict[str, list[str]] = {}
        for vm in lab.vms:
            by_folder.setdefault(vm.folder, []).append(vm.name)
        self.assertEqual(len(by_folder["lab/Prod"]), 3)
        self.assertEqual(by_folder["lab"], ["freebsd13-nas"])

    def test_subtree_membership(self):
        """A subtree match on lab/Prod must include every VM in its
        child folders — the shape preflight's --folder relies on."""
        lab = config.load_lab(LAB)
        under_prod = sorted(
            vm.name
            for vm in lab.vms
            if vm.folder == "lab/Prod" or vm.folder.startswith("lab/Prod/")
        )
        self.assertEqual(
            under_prod,
            [
                "debian12-web",
                "photon-container",
                "rhel9-db",
                "sap-hana-production-database-server-primary-001-eu-central-instance",
                "sles15-sap",
                "solaris11-db",
                "ubuntu-k8s-node",
                "ubuntu-vgpu-vdi",
                "win10-legacy-desk",
                "win10-vdi",
                "win11-amd-mxgpu",
                "win11-secure",
                "win2019-fileserver",
                "win2022-sql",
            ],
        )

    def test_templates_and_appliances_still_grouped(self):
        lab = config.load_lab(LAB)
        templates = [v for v in lab.vms if v.template]
        self.assertEqual(len(templates), 3)
        self.assertTrue(all(v.folder == "lab/Templates" for v in templates))
        appliances = [v for v in lab.vms if v.vapp is not None]
        self.assertEqual(len(appliances), 2)
        self.assertTrue(all(v.folder == "lab/Appliances" for v in appliances))
        self.assertTrue(all(v.vapp_container == "lab-appliances" for v in appliances))

    def test_declared_assignments_carry_folder(self):
        declared = config.load_declared_assignments(LAB)
        self.assertEqual(declared["solaris11-db"]["folder"], "lab/Prod/DB")
        self.assertEqual(declared["win2022-template"]["folder"], "lab/Templates")


class TestFindFolderPath(unittest.TestCase):
    def test_nested_lookup(self):
        dc, vm_folder = _dc_mock()
        build.ensure_folder(dc, "lab/Prod/Web")
        found = vcenter.find_folder_path(dc, "lab/Prod/Web")
        self.assertIsNotNone(found)
        self.assertEqual(found.name, "Web")
        self.assertEqual(vcenter.find_folder_path(dc, "lab/Prod").name, "Prod")
        self.assertEqual(vcenter.find_folder_path(dc, "lab").name, "lab")

    def test_root_path_is_vm_folder(self):
        dc, vm_folder = _dc_mock()
        self.assertIs(vcenter.find_folder_path(dc, ""), vm_folder)

    def test_missing_segment_returns_none(self):
        dc, _ = _dc_mock()
        build.ensure_folder(dc, "lab/Prod")
        self.assertIsNone(vcenter.find_folder_path(dc, "lab/Prod/Web"))
        self.assertIsNone(vcenter.find_folder_path(dc, "nope"))
        self.assertIsNone(vcenter.find_folder_path(dc, "nope/Prod"))

    def test_find_vm_folder_handles_paths(self):
        dc, _ = _dc_mock()
        build.ensure_folder(dc, "lab/Prod/Web")
        self.assertEqual(vcenter.find_vm_folder(dc, "lab/Prod/Web").name, "Web")
        self.assertIsNone(vcenter.find_vm_folder(dc, "lab/Nope"))


class TestEnsureFolder(unittest.TestCase):
    def test_creates_missing_parents_recursively(self):
        dc, vm_folder = _dc_mock()
        leaf = build.ensure_folder(dc, "lab/Prod/Web")
        self.assertEqual(leaf.name, "Web")
        self.assertEqual(vm_folder.CreateFolder.call_count, 1)
        lab = vcenter.find_folder_path(dc, "lab")
        self.assertEqual(lab.CreateFolder.call_count, 1)
        prod = vcenter.find_folder_path(dc, "lab/Prod")
        self.assertEqual(prod.CreateFolder.call_count, 1)

    def test_second_run_creates_nothing(self):
        dc, _ = _dc_mock()
        build.ensure_folder(dc, "lab/Prod/Web")
        build.ensure_folder(dc, "lab/Prod/DB")
        counts_before = {
            f.name: f.CreateFolder.call_count
            for f in (dc.vmFolder, vcenter.find_folder_path(dc, "lab"))
        }
        build.ensure_folder(dc, "lab/Prod/Web")
        build.ensure_folder(dc, "lab/Prod/DB")
        counts_after = {
            f.name: f.CreateFolder.call_count
            for f in (dc.vmFolder, vcenter.find_folder_path(dc, "lab"))
        }
        self.assertEqual(counts_before, counts_after)

    def test_shared_parent_created_once(self):
        dc, vm_folder = _dc_mock()
        build.ensure_folder(dc, "lab/Prod/Web")
        build.ensure_folder(dc, "lab/Prod/DB")
        # One "lab" child, one "Prod" child — no duplicates.
        self.assertEqual([c.name for c in vm_folder.childEntity], ["lab"])
        lab = vcenter.find_folder_path(dc, "lab")
        self.assertEqual([c.name for c in lab.childEntity], ["Prod"])
        prod = vcenter.find_folder_path(dc, "lab/Prod")
        self.assertEqual(sorted(c.name for c in prod.childEntity), ["DB", "Web"])

    def test_empty_path_is_vm_folder(self):
        dc, vm_folder = _dc_mock()
        self.assertIs(build.ensure_folder(dc, ""), vm_folder)
        vm_folder.CreateFolder.assert_not_called()

    def test_flat_name_still_works(self):
        dc, vm_folder = _dc_mock()
        folder = build.ensure_folder(dc, "lab-dev")
        self.assertEqual(folder.name, "lab-dev")
        self.assertEqual([c.name for c in vm_folder.childEntity], ["lab-dev"])


class TestFolderRelPath(unittest.TestCase):
    def _chain(self):
        dc, vm_folder = _dc_mock()
        lab = _folder_mock("lab", parent=vm_folder)
        prod = _folder_mock("Prod", parent=lab)
        web = _folder_mock("Web", parent=prod)
        return dc, vm_folder, lab, prod, web

    def test_nested_path(self):
        _, _, _, _, web = self._chain()
        self.assertEqual(vcenter.folder_rel_path(web), "lab/Prod/Web")

    def test_vm_folder_is_empty_path(self):
        _, vm_folder, _, _, _ = self._chain()
        self.assertEqual(vcenter.folder_rel_path(vm_folder), "")

    def test_top_level_folder(self):
        _, _, lab, _, _ = self._chain()
        self.assertEqual(vcenter.folder_rel_path(lab), "lab")

    def test_cache_is_shared(self):
        _, _, _, prod, web = self._chain()
        cache: dict = {}
        self.assertEqual(vcenter.folder_rel_path(web, cache), "lab/Prod/Web")
        # Every link in the chain is cached by MoRef id, so a second
        # VM under the same folder walks nothing.
        self.assertEqual(len(cache), 4)  # vm, lab, Prod, Web
        self.assertEqual(vcenter.folder_rel_path(prod, cache), "lab/Prod")

    def test_unreadable_parent_is_none(self):
        class _UnreadableFolder:
            # A folder whose parent read raises (e.g. a stale MoRef):
            # the walk must report "unknown", not crash. folder_rel_path
            # only touches _moId/.parent/.name on the folder itself, so
            # a plain stub works here (no vim.Folder isinstance needed).
            _moId = "folder-unreadable"

            @property
            def parent(self):
                raise RuntimeError("gone")

            @property
            def name(self):
                return "unreadable"

        self.assertIsNone(vcenter.folder_rel_path(_UnreadableFolder()))

    def test_chain_not_reaching_a_datacenter_is_none(self):
        orphan = _folder_mock("orphan", parent=object())
        self.assertIsNone(vcenter.folder_rel_path(orphan))


class TestVerifyFolder(unittest.TestCase):
    def test_matching_folder_passes(self):
        failures, warnings = verify.verify_vm(_vm_spec(), _live())
        self.assertEqual(failures, [])
        self.assertEqual(warnings, [])

    def test_mismatched_folder_fails(self):
        failures, _ = verify.verify_vm(_vm_spec(), _live(folder="lab/Lab/GPU"))
        self.assertEqual(len(failures), 1)
        self.assertIn("folder", failures[0])

    def test_unknown_folder_skips(self):
        failures, warnings = verify.verify_vm(_vm_spec(), _live(folder=None))
        self.assertEqual(failures, [])
        self.assertEqual(warnings, [])

    def test_root_folder_matches_empty_spec(self):
        spec = _vm_spec(folder="")
        failures, warnings = verify.verify_vm(spec, _live(folder=""))
        self.assertEqual(failures, [])
        self.assertEqual(warnings, [])


class TestSummarizeFolder(unittest.TestCase):
    def _runtime(self, **overrides):
        base = {
            "power": "poweredOff",
            "cbt": True,
            "boot_delay_ms": 3000,
            "tools_status": "toolsNotInstalled",
            "snapshots": [],
            "current_snapshot": None,
            "folder": "lab/Prod/Web",
        }
        base.update(overrides)
        return base

    def test_summarize_vm_carries_folder(self):
        summary = query.summarize_vm(
            _mock_vm(),
            runtime=self._runtime(),
            declared={"tools_status": "toolsOk", "folder": "lab/Prod/Web"},
        )
        self.assertEqual(summary["folder"], "lab/Prod/Web")
        self.assertIs(summary["folder_mismatch"], False)

    def test_summarize_vm_flags_folder_mismatch(self):
        summary = query.summarize_vm(
            _mock_vm(),
            runtime=self._runtime(folder="lab/Lab/GPU"),
            declared={"tools_status": "toolsOk", "folder": "lab/Prod/Web"},
        )
        self.assertIs(summary["folder_mismatch"], True)

    def test_summarize_vm_unknown_folder_stays_none(self):
        summary = query.summarize_vm(
            _mock_vm(),
            runtime=self._runtime(folder=None),
            declared={"tools_status": "toolsOk", "folder": "lab/Prod/Web"},
        )
        self.assertIsNone(summary["folder"])
        self.assertNotIn("folder_mismatch", summary)

    def test_summarize_vm_without_leaf_data_has_no_folder(self):
        summary = query.summarize_vm(_mock_vm())
        self.assertIsNone(summary["folder"])


if __name__ == "__main__":
    unittest.main()
