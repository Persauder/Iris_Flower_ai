from pathlib import Path
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # дозволяє читати частково биті файли

root = Path("dataset")
bad_dir = root / "_bad"
bad_dir.mkdir(exist_ok=True, parents=True)

exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

bad = []
for p in root.rglob("*"):
    if p.is_file() and p.suffix.lower() in exts:
        try:
            with Image.open(p) as im:
                im.verify()  # швидка перевірка цілісності
        except Exception as e:
            print("BAD:", p, "|", e)
            bad.append(p)

print(f"Found {len(bad)} bad files")
for p in bad:
    target = bad_dir / p.name
    try:
        p.rename(target)
    except Exception:
        # якщо на інший диск — скопіюй вручну, а файл видали
        pass
