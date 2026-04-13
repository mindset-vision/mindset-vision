"""generate an HTML quality-check report for all mindset-vision generators."""
import base64
import csv
import importlib
import inspect
import io
import random
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml
from PIL import Image

SAMPLE_CAP = 6


def cap_sample_params(gen_fn, cap=SAMPLE_CAP):
    """inspect gen_fn, return kwargs capping all num_samples* params to cap."""
    spec = inspect.getfullargspec(gen_fn)
    defaults = dict(zip(reversed(spec.args or []), reversed(spec.defaults or [])))
    return {k: min(v, cap) for k, v in defaults.items() if k.startswith("num_samples") and isinstance(v, int)}


def sample_balanced(annotation_csv, output_folder, target=12):
    """read annotation.csv, group by condition (first path component), return balanced image paths."""
    by_condition = defaultdict(list)
    with open(annotation_csv) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            rel = Path(row[0])
            img_path = output_folder / rel
            if not img_path.exists():
                found = list(output_folder.rglob(rel.name))
                img_path = found[0] if found else None
            if img_path is None or not img_path.exists():
                continue
            condition = img_path.parent.name if img_path.parent != output_folder else "all"
            by_condition[condition].append(img_path)

    conditions = sorted(by_condition.keys())
    n = len(conditions)
    if n == 0:
        return [], []

    base, remainder = divmod(target, n)
    selected = []
    for i, cond in enumerate(conditions):
        quota = base + (1 if i < remainder else 0)
        pool = by_condition[cond]
        selected.extend(random.sample(pool, min(quota, len(pool))))

    shortfall = target - len(selected)
    if shortfall > 0:
        used = set(selected)
        remaining = [p for paths in by_condition.values() for p in paths if p not in used]
        selected.extend(random.sample(remaining, min(shortfall, len(remaining))))

    result = selected[:target]
    random.shuffle(result)
    return result, conditions


class ReportGenerator:
    """generates an HTML quality-check page with sidebar nav and balanced grids."""

    def __init__(self, config_path):
        """load config from yaml."""
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.cols = self.config["grid"]["cols"]
        self.rows = self.config["grid"]["rows"]
        self.target = self.cols * self.rows
        self.shared = self.config.get("shared", {})

    def _generate_and_sample(self, cfg, tmp_dir, name):
        """generate with capped params, then balanced-sample from annotation.csv."""
        output_dir = Path(tmp_dir) / name

        from mindset.cli import _load_registry
        from mindset.generators import get_generator
        _load_registry()

        info = get_generator(cfg["registry_name"])
        gen_fn = info["func"]

        capped = cap_sample_params(gen_fn, SAMPLE_CAP)
        params = {**self.shared, **cfg.get("overrides", {}), **capped, "output_folder": str(output_dir)}
        gen_fn(**params)

        ann = output_dir / "annotation.csv"
        if ann.exists():
            return sample_balanced(ann, output_dir, self.target)
        images = sorted(output_dir.rglob("*.png"))[:self.target]
        conditions = list({p.parent.name for p in images})
        return images, conditions

    def _to_base64(self, path):
        """compress image to base64 jpeg."""
        img = Image.open(path).convert("RGB")
        img.thumbnail((224, 224), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode()

    def _render_section(self, label, image_paths, conditions):
        """render one generator section as an HTML grid."""
        cond_counts = defaultdict(int)
        for p in image_paths:
            cond_counts[p.parent.name] += 1
        cond_summary = ", ".join(f"{k} ({v})" for k, v in sorted(cond_counts.items()))

        html = f'<h2>{label}</h2>\n'
        html += f'<p class="meta">{len(image_paths)} samples | {len(conditions)} conditions: {cond_summary}</p>\n'
        html += '<div class="grid">\n'
        for path in image_paths:
            html += f'  <div class="cell"><img src="data:image/jpeg;base64,{self._to_base64(path)}" alt="{path.parent.name}"><span>{path.parent.name}</span></div>\n'
        html += '</div>\n'
        return html

    def generate(self, output_dir):
        """generate the full HTML report."""
        generators = self.config["generators"]
        total = len(generators)
        sections = []
        all_conditions = 0

        with tempfile.TemporaryDirectory() as tmp_dir:
            for idx, (name, cfg) in enumerate(generators.items(), 1):
                label = cfg.get("label", name)
                print(f"  [{idx}/{total}] {label}", end="", flush=True)
                try:
                    images, conditions = self._generate_and_sample(cfg, tmp_dir, name)
                    all_conditions += len(conditions)
                    sections.append((name, label, self._render_section(label, images, conditions), len(conditions)))
                    print(f" ({len(images)} images, {len(conditions)} conditions)")
                except Exception as e:
                    print(f" - SKIPPED ({e.__class__.__name__}: {e})")

        date_str = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        rendered = len(sections)

        nav_html = ""
        current_cat = ""
        for name, label, _, n_cond in sections:
            parts = generators[name].get("module", "").split(".")
            cat = parts[2] if len(parts) > 2 else "other"
            if cat != current_cat:
                nav_html += f'<div class="nav-cat">{cat.replace("_", " ")}</div>\n'
                current_cat = cat
            nav_html += f'<a href="#{name}" class="nav-link">{label} <span class="badge">{n_cond}</span></a>\n'

        body_html = "".join(f'<section id="{name}">{html}</section>\n' for name, _, html, _ in sections)

        page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>mindset-vision quality check</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, sans-serif; background: #fff; color: #1f2328; display: flex; min-height: 100vh; }}

  .sidebar {{ position: fixed; top: 0; left: 0; width: 240px; height: 100vh; overflow-y: auto; background: #f6f8fa; border-right: 1px solid #d1d9e0; padding: 16px 0; }}
  .sidebar-header {{ padding: 0 16px 14px; border-bottom: 1px solid #d1d9e0; margin-bottom: 8px; }}
  .sidebar-header strong {{ font-size: 0.9em; }}
  .sidebar-header .stats {{ font-size: 0.7em; color: #656d76; margin-top: 4px; }}
  .nav-cat {{ font-size: 0.68em; text-transform: uppercase; letter-spacing: 0.04em; color: #656d76; padding: 14px 16px 4px; font-weight: 600; }}
  .nav-link {{ display: flex; justify-content: space-between; align-items: center; padding: 4px 16px 4px 20px; color: #1f2328; text-decoration: none; font-size: 0.8em; border-left: 2px solid transparent; }}
  .nav-link:hover {{ background: #eaeef2; border-left-color: #0969da; color: #0969da; }}
  .badge {{ font-size: 0.7em; color: #656d76; background: #e8ecf0; padding: 1px 5px; border-radius: 8px; }}

  .main {{ margin-left: 240px; flex: 1; padding: 32px 48px; max-width: 920px; }}
  h1 {{ font-size: 1.4em; font-weight: 600; margin-bottom: 2px; }}
  .subtitle {{ color: #656d76; font-size: 0.85em; margin-bottom: 36px; }}

  section {{ margin-bottom: 44px; }}
  h2 {{ font-size: 1.05em; font-weight: 600; padding-bottom: 6px; border-bottom: 1px solid #d1d9e0; margin-bottom: 6px; }}
  .meta {{ color: #656d76; font-size: 0.75em; margin-bottom: 8px; }}

  .grid {{ display: grid; grid-template-columns: repeat({self.cols}, 1fr); gap: 5px; }}
  .cell {{ text-align: center; }}
  .cell img {{ width: 100%; border-radius: 3px; border: 1px solid #d1d9e0; }}
  .cell span {{ display: block; font-size: 0.6em; color: #656d76; margin-top: 1px; }}

  footer {{ margin-top: 48px; padding-top: 12px; border-top: 1px solid #d1d9e0; color: #656d76; font-size: 0.7em; }}
  @media (max-width: 768px) {{ .sidebar {{ display: none; }} .main {{ margin-left: 0; padding: 16px; }} }}
</style>
</head>
<body>
<nav class="sidebar">
  <div class="sidebar-header">
    <strong>mindset-vision</strong>
    <div class="stats">{rendered}/{total} generators<br>{all_conditions} total conditions</div>
  </div>
  {nav_html}
</nav>
<div class="main">
  <h1>quality check</h1>
  <p class="subtitle">{date_str} | {rendered}/{total} generators | {all_conditions} total conditions | {self.target} balanced samples each</p>
  {body_html}
  <footer>auto-generated by quality-check workflow</footer>
</div>
</body>
</html>"""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "index.html"
        out_path.write_text(page)

        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"\nHTML report: {out_path} ({size_mb:.1f} MB)")
        if size_mb > self.config["report"]["max_size_mb"]:
            print(f"warning: exceeds {self.config['report']['max_size_mb']} MB limit")
            sys.exit(1)


if __name__ == "__main__":
    config_path = Path(__file__).parent / "config.yaml"
    output_dir = Path(tempfile.gettempdir()) / "mindset-report-site"
    ReportGenerator(config_path).generate(str(output_dir))
