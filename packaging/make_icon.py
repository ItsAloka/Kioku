"""Draw the app icon (記 on a violet rounded tile) → src/kioku/resources/app.ico + app.png."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "src" / "kioku" / "resources"
FONTS = ["C:/Windows/Fonts/YuGothB.ttc", "C:/Windows/Fonts/meiryob.ttc", "C:/Windows/Fonts/msgothic.ttc"]


def draw(size: int = 1024) -> Image.Image:
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    top, bottom = (155, 123, 255), (96, 64, 214)
    grad = Image.new("RGBA", (size, size))
    for y in range(size):
        t = y / (size - 1)
        grad.paste(tuple(round(a + (b - a) * t) for a, b in zip(top, bottom)) + (255,), (0, y, size, y + 1))
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size - 1, size - 1), radius=size // 4.6, fill=255)
    im.paste(grad, (0, 0), mask)
    font = next(ImageFont.truetype(f, int(size * 0.62)) for f in FONTS if Path(f).is_file())
    d = ImageDraw.Draw(im)
    d.text((size / 2, size / 2 + size * 0.02), "記", font=font, fill="white", anchor="mm")
    return im


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    big = draw()
    big.resize((256, 256), Image.Resampling.LANCZOS).save(OUT / "app.png")
    big.save(OUT / "app.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("wrote", OUT / "app.ico")
