"""Лёгкие модули импортируются без torch и OpenCV: их используют events/verdict на машине без GPU."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

LIGHT_MODULES = (
    "worker.perception",
    "worker.perception.geometry",
    "worker.perception.tracking",
    "worker.perception.zones",
    "worker.perception.states",
    "worker.perception.models",
    "worker.scene",
    "worker.scene.contract",
    "worker.scene.builder",
)


class LightImportsTest(unittest.TestCase):
    def test_no_heavy_dependencies_on_import(self) -> None:
        code = (
            "import importlib, sys\n"
            f"for name in {LIGHT_MODULES!r}:\n"
            "    importlib.import_module(name)\n"
            "print(sorted(m for m in ('torch', 'transformers', 'cv2', 'numpy') if m in sys.modules))\n"
        )
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "[]")


if __name__ == "__main__":
    unittest.main()
