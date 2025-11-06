from pathlib import Path, PurePath

from dvclive.utils import isinstance_without_import

from .base import Data


class Image(Data):
    suffixes = (".jpg", ".jpeg", ".gif", ".png")
    subfolder = "images"

    @property
    def output_path(self) -> Path:
        _path = self.output_folder / self.name
        _path.parent.mkdir(exist_ok=True, parents=True)
        return _path

    @staticmethod
    def could_log(val: object) -> bool:
        acceptable = {
            ("numpy", "ndarray"),
            ("matplotlib.figure", "Figure"),
            ("PIL.Image", "Image"),
        }
        for cls in type(val).mro():
            if any(isinstance_without_import(val, *cls) for cls in acceptable):
                return True
        return isinstance(val, (PurePath, str))

    def dump(self, val, **kwargs) -> None:  # noqa: ARG002
        if isinstance_without_import(val, "numpy", "ndarray"):
            from PIL import Image as ImagePIL

            ImagePIL.fromarray(val).save(self.output_path)
        elif isinstance_without_import(val, "matplotlib.figure", "Figure"):
            import matplotlib.pyplot as plt

            val.savefig(self.output_path)
            plt.close(val)
        elif isinstance_without_import(val, "PIL.Image", "Image"):
            val.save(self.output_path)
