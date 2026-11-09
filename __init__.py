from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

from .magik_image_resize import MagikImageResize


class MagikImageResizeExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [MagikImageResize]


async def comfy_entrypoint() -> MagikImageResizeExtension:
    return MagikImageResizeExtension()
