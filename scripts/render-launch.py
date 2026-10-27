"""Build launch videos from captured UI and actual Genesis footage. macOS + FFmpeg."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "artifacts/media"
OUT = ROOT / "artifacts/release"
OUT.mkdir(parents=True, exist_ok=True)
FONT = "/System/Library/Fonts/Hiragino Sans GB.ttc"
MONO = "/System/Library/Fonts/Menlo.ttc"
SCENES = json.loads((ROOT / "docs/media/storyboard.json").read_text())
INK, MUTED, GREEN = "#172624", "#566b66", "#087f70"
SIZE = (1920, 1080)


def run(*args):
    subprocess.run([str(a) for a in args], check=True)


def probe(file):
    return json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(file)]))


def text(draw, value, x, y, size=42, fill=INK, width=1740, center=False, mono=False):
    font = ImageFont.truetype(MONO if mono else FONT, size)
    lines, line = [], ""
    for char in value:
        if char == "\n" or (line and draw.textlength(line + char, font=font) > width):
            lines.append(line)
            line = "" if char == "\n" else char
        else:
            line += char
    if line:
        lines.append(line)
    for line in lines:
        px = x + (width - draw.textlength(line, font=font)) / 2 if center else x
        draw.text((px, y), line, font=font, fill=fill)
        y += int(size * 1.4)
    return y


def slide(kind):
    image = Image.new("RGB", SIZE, "#edf3f2")
    draw = ImageDraw.Draw(image)
    if kind == "package":
        text(draw, "A REPRODUCIBLE SCENE PACKAGE", 110, 180, 38, GREEN)
        rows = [("assets/model.glb", "MIT procedural mesh"), ("manifest.json", "Gravity + solver parameters"), ("flex-profile.json", "Current material preset"), ("genesis_scene.py", "Runnable Genesis entry point"), ("requirements.txt", "Pinned direct dependencies")]
        for i, (name, detail) in enumerate(rows):
            y = 290 + i * 100
            draw.line((110, y + 78, 1800, y + 78), fill="#cad8d4", width=2)
            text(draw, name, 110, y, 40, mono=True, width=800)
            text(draw, detail, 980, y, 34, MUTED, width=810)
    elif kind == "metrics":
        metrics = json.loads((MEDIA / "metrics.json").read_text())
        rows = [f"Genesis {metrics['genesis_version']} / CPU", f"{metrics['steps']} steps  x  {metrics['dt']:.3f} s", f"{metrics['vertex_count']} simulated vertices", f"{metrics['max_shape_deformation_m'] * 100:.1f} cm max shape deformation", "Finite trajectory / ground contact"]
        text(draw, "MEASURED IN THIS SAMPLE RUN", 110, 180, 38, GREEN)
        for i, value in enumerate(rows):
            text(draw, value, 110, 290 + i * 100, 47)
        text(draw, "Not a material calibration or real-world accuracy benchmark.", 110, 830, 30, MUTED)
    elif kind == "limits":
        text(draw, "v0.1.0 ALPHA", 110, 180, 40, GREEN)
        rows = ["Local single-user workbench", "Optional DashScope / Tripo service fees", "Known Python dependency advisories", "Trusted sample only, in an isolated environment", "Cloth export is experimental"]
        for i, value in enumerate(rows):
            text(draw, value, 110, 290 + i * 100, 43)
    else:
        text(draw, "Real2Sim Studio", 110, 190, 100)
        text(draw, "MIT-LICENSED WORKBENCH + PROCEDURAL SAMPLE", 115, 355, 31, GREEN)
        text(draw, "npm ci\nnpm run demo", 115, 455, 53, mono=True)
        text(draw, "github.com/eust-w/real2sim-platform", 115, 690, 48, GREEN)
        text(draw, "v0.1.0 Alpha  |  See README and SECURITY.md before running.", 115, 810, 29, MUTED)
    file = MEDIA / f"slide-{kind}.png"
    image.save(file)
    return file


def overlay(index, scene, lang, vertical=False):
    width, height = (1080, 1920) if vertical else SIZE
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    header, footer = (260, 390) if vertical else (132, 184)
    draw.rectangle((0, 0, width, header), fill="#f6f9f8")
    draw.rectangle((0, height - footer, width, height), fill="#172624")
    title = scene["title"] if lang == "zh" else scene["enTitle"]
    text(draw, "Real2Sim Studio / v0.1.0 Alpha", 48, 24, 25, GREEN, width=width-96)
    text(draw, title, 48, 77, 48 if vertical else 46, width=width - 96)
    caption = scene[lang]
    text(draw, caption, 60, height - footer + 30, 42 if vertical else 36, "#ffffff", width=width-120, center=True)
    label = "Genesis FEM / CPU / 0.2x replay" if scene["source"] == "physics" else "Actual workbench capture" if scene["source"] in ("ui", "settings") else "Reproducible Alpha / see repository"
    text(draw, label, 48, height - footer - 43, 23, INK, width=width-96)
    text(draw, "Synthetic narration / no cloned voice" if lang == "en" else "合成配音 · 非真人仿声", 48, height-55, 23, "#b5ccc5", width=width-96)
    draw.rectangle((0, height - 7, int(width * (index + 1) / (5 if vertical else 10)), height), fill="#24bfa8")
    file = MEDIA / f"overlay-{lang}-{'v' if vertical else 'h'}-{index}.png"
    image.save(file)
    return file


def audio(index, scene):
    file = MEDIA / f"voice-{index}.aiff"
    if not file.exists():
        run("say", "-v", "Tingting", "-r", "190", "-o", file, scene["zh"])
    return file


def render(lang="zh", vertical=False):
    selection = [0, 2, 4, 6, 9] if vertical else list(range(10))
    segments, subtitles, cursor = [], [], 0
    for out_index, index in enumerate(selection):
        scene = SCENES[index]
        duration = scene["duration"]
        voice = audio(index, scene)
        voice_duration = float(probe(voice)["format"]["duration"])
        tempo = max(1, voice_duration / (duration - 0.7))
        source = scene["source"]
        inputs = []
        if source == "physics":
            inputs = ["-stream_loop", "-1", "-i", MEDIA / "simulation.mp4"]
            transform = "setpts=5*PTS,crop=520:640:380:40,scale=1080:1330,pad=1080:1920:0:270:color=0xedf3f2" if vertical else "setpts=5*PTS,crop=1000:500:140:140,scale=1720:860,pad=1920:1080:100:132:color=0xedf3f2"
        elif source in ("ui", "settings"):
            file = MEDIA / ("mobile.webm" if vertical else "workbench.webm")
            inputs = ["-stream_loop", "-1", "-ss", "10" if source == "settings" else "4", "-i", file]
            transform = "scale=900:1600,pad=1080:1920:90:200:color=0xedf3f2" if vertical else "scale=1568:882,pad=1920:1080:176:132:color=0xedf3f2"
        else:
            file = slide(source)
            inputs = ["-loop", "1", "-framerate", "30", "-i", file]
            transform = "scale=1080:608,pad=1080:1920:0:520:color=0xedf3f2" if vertical else "null"
        cover = overlay(out_index, scene, lang, vertical)
        output = MEDIA / f"segment-{lang}-{'v' if vertical else 'h'}-{out_index}.mp4"
        run("ffmpeg", "-y", "-loglevel", "error", *inputs, "-loop", "1", "-i", cover, "-i", voice,
            "-filter_complex", f"[0:v]{transform},setsar=1,fps=30[base];[base][1:v]overlay=0:0,format=yuv420p[v];[2:a]atempo={tempo:.5f},apad,atrim=duration={duration},loudnorm=I=-16:TP=-1.5:LRA=11[a]",
            "-map", "[v]", "-map", "[a]", "-t", duration, "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-threads", "4", "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-movflags", "+faststart", output)
        segments.append(output)
        subtitles.append((cursor, cursor + duration, scene[lang]))
        cursor += duration
        print(f"Rendered {lang} {'vertical' if vertical else 'landscape'} {out_index+1}/{len(selection)}", flush=True)
    name = f"real2sim-{'short' if vertical else 'intro'}-{lang}"
    listing = MEDIA / f"{name}-concat.txt"
    listing.write_text("".join(f"file '{p}'\n" for p in segments))
    run("ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", listing, "-c", "copy", "-movflags", "+faststart", OUT / f"{name}.mp4")
    def timestamp(seconds):
        return f"00:{seconds // 60:02}:{seconds % 60:02},000"
    (OUT / f"{name}.srt").write_text("\n\n".join(f"{i+1}\n{timestamp(a)} --> {timestamp(b)}\n{caption}" for i, (a, b, caption) in enumerate(subtitles)) + "\n")
    run("ffmpeg", "-y", "-loglevel", "error", "-ss", "2", "-i", OUT / f"{name}.mp4", "-frames:v", "1", OUT / f"{name}-cover.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=["zh", "en", "all"], default="all")
    parser.add_argument("--vertical", action="store_true")
    args = parser.parse_args()
    for language in (["zh", "en"] if args.language == "all" else [args.language]):
        render(language, args.vertical)
    shutil.copyfile(ROOT / "docs/media/storyboard.json", OUT / "storyboard.json")
