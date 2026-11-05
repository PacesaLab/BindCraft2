"""Include the existing preset tree and its referenced structures in wheels."""

from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPy(build_py):
    def run(self):
        super().run()
        root = Path(__file__).parent
        for name in ("settings", "scaffolds"):
            self.copy_tree(
                str(root / name),
                str(Path(self.build_lib) / "bindcraft" / "_resources" / name),
            )


setup(cmdclass={"build_py": BuildPy})
