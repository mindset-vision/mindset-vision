"""generate a PDF report of illusion samples from config.yaml."""
import importlib
import inspect
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


class IllusionReportGenerator:
    """generates a multi-page PDF of illusion samples."""

    def __init__(self, config_path):
        """load config from yaml."""
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.grid_cols = self.config["grid"]["cols"]
        self.grid_rows = self.config["grid"]["rows"]
        self.samples_per_page = self.grid_cols * self.grid_rows
        self.dpi = self.config["pdf"]["dpi"]
        self.max_size_mb = self.config["pdf"]["max_size_mb"]
        self.shared = self.config.get("shared", {})

    def _generate_samples(self, name, illusion_cfg, tmp_dir):
        """generate samples for one illusion, return list of image paths."""
        output_dir = Path(tmp_dir) / name
        module = importlib.import_module(illusion_cfg["legacy_path"])
        gen_fn = module.generate_all

        accepted = set(inspect.getfullargspec(gen_fn).args)
        params = {**self.shared, **illusion_cfg.get("params", {}), "output_folder": str(output_dir)}
        filtered = {k: v for k, v in params.items() if k in accepted}

        gen_fn(**filtered)
        return sorted(output_dir.rglob("*.png"))[:self.samples_per_page]

    def _render_page(self, pdf, label, image_paths, page_num, total_pages):
        """render one page of the PDF with a grid of images."""
        fig, axes = plt.subplots(self.grid_rows, self.grid_cols, figsize=(8.27, 11.69), dpi=self.dpi)
        fig.suptitle(label, fontsize=14, fontweight="bold", y=0.97)
        axes = axes.flatten()

        for idx, ax in enumerate(axes):
            ax.axis("off")
            if idx < len(image_paths):
                img = Image.open(image_paths[idx])
                ax.imshow(img)
                ax.set_title(image_paths[idx].parent.name, fontsize=6, color="gray")

        fig.text(0.5, 0.01, f"page {page_num} / {total_pages}", ha="center", fontsize=8, color="gray")
        fig.tight_layout(rect=[0, 0.02, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

    def _compress_pdf(self, input_path, output_path):
        """compress PDF using ghostscript if available, else copy as-is."""
        gs = shutil.which("gs")
        if gs is None:
            shutil.copy(input_path, output_path)
            return
        subprocess.run([
            gs, "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.4",
            "-dPDFSETTINGS=/ebook", "-dNOPAUSE", "-dBATCH", "-dQUIET",
            f"-sOutputFile={output_path}", str(input_path),
        ], check=True)

    def generate(self, output_path):
        """generate the full PDF report."""
        illusions = self.config["illusions"]
        total_pages = len(illusions)

        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_pdf = Path(tmp_dir) / "raw.pdf"

            with PdfPages(str(raw_pdf)) as pdf:
                rendered = 0
                for page_num, (name, cfg) in enumerate(illusions.items(), 1):
                    label = cfg.get("label", name)
                    print(f"  [{page_num}/{total_pages}] {label}", end="")
                    try:
                        image_paths = self._generate_samples(name, cfg, tmp_dir)
                        rendered += 1
                        print()
                        self._render_page(pdf, label, image_paths, rendered, total_pages)
                    except Exception as e:
                        print(f" - SKIPPED ({e.__class__.__name__}: {e})")

            self._compress_pdf(raw_pdf, output_path)

        size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"\nPDF saved: {output_path} ({size_mb:.1f} MB)")

        if size_mb > self.max_size_mb:
            print(f"warning: PDF exceeds {self.max_size_mb} MB limit ({size_mb:.1f} MB)")
            sys.exit(1)


if __name__ == "__main__":
    config_path = Path(__file__).parent / "config.yaml"
    date_str = datetime.now().strftime("%Y-%m-%d")
    output = Path(tempfile.gettempdir()) / f"mindset-illusion-report-{date_str}.pdf"

    gen = IllusionReportGenerator(config_path)
    gen.generate(str(output))
    print(f"\nreport at: {output}")
