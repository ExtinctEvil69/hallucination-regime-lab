"""Record the real UI (replay of chart 20003) as animated GIFs: frame i = headless Chrome state at virtual time t_i."""
import subprocess, sys
from pathlib import Path
from PIL import Image
D = Path(__file__).parent; F = D / "frames"; F.mkdir(exist_ok=True)
C = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
times = [500] + list(range(2000, 44001, 1500))
for i, t in enumerate(times):
    out = F / f"f{i:03d}.png"
    if not out.exists():
        subprocess.run([C, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--window-size=1480,1950",
                        f"--virtual-time-budget={t}", f"--user-data-dir={F}/prof{i}", f"--screenshot={out}",
                        "http://localhost:8765/#replay&autoplay"], capture_output=True, timeout=180)
    print(i, t, out.exists(), flush=True)
def gif(box, name, width):
    frames = []
    for i in range(len(times)):
        im = Image.open(F / f"f{i:03d}.png").convert("RGB").crop(box)
        im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)
        frames.append(im.quantize(colors=200, method=Image.Quantize.MEDIANCUT))
    frames[0].save(D / "figures" / name, save_all=True, append_images=frames[1:], duration=[900] * (len(frames) - 1) + [3500],
                   loop=0, optimize=True)
    frames[-1].convert("RGB").save(D / "figures" / name.replace(".gif", "_final.png"))
gif((0, 60, 1480, 1080), "demo_conversation.gif", 1200)
gif((0, 1080, 1480, 1840), "demo_markov_chain.gif", 1200)
print("done", [p.stat().st_size // 1024 for p in (D / "figures").glob("demo_*.gif")])
