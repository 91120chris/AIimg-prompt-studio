from pathlib import Path

from PIL import Image


def image_dimensions(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.width, image.height


def generate_thumbnail(source_path: Path, thumbnail_path: Path, max_side: int = 512) -> Path | None:
    try:
        thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source_path) as image:
            image.thumbnail((max_side, max_side))
            if image.mode not in {"RGB", "RGBA"}:
                image = image.convert("RGB")
            image.save(thumbnail_path, format="WEBP", quality=86)
    except Exception:
        return None
    return thumbnail_path
