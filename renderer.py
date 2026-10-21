"""Agent 4: renders each retrosynthesis route as an RDKit PNG for chemists."""

from pathlib import Path

import dspy
from PIL import Image, ImageDraw
from rdkit import Chem
from rdkit.Chem import Draw

from config import RendererConfig


class RouteRenderer(dspy.Module):
    """Agent 4: draws each route as one PNG, one row per reaction step, target at the top."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or RendererConfig()
        self.out_dir = Path(self.config.out_dir)
        self.sub_img_size = (self.config.width, self.config.height)

    def forward(self, ranked):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, rr in enumerate(ranked, 1):
            rows = [self._draw_step(step) for step in rr.root.steps()]
            if not rows:
                continue
            status = "valid" if rr.valid else "INVALID"
            header = [f"Route {i} | score {rr.score:.3f} | {rr.n_steps} step(s) | {status}"]
            header += rr.issues[:3]
            path = self.out_dir / f"route_{i:02d}.png"
            self._stack(rows, header).save(path)
            paths.append(path)
        return paths

    def _draw_step(self, step):
        mols, legends = [], []
        for child in step.children:
            mols.append(Chem.MolFromSmiles(child.smiles))
            legends.append("" if child.children else "in stock")
        mols.append(Chem.MolFromSmiles(step.smiles))
        legends.append(f"confidence {step.step_prob:.2f}")
        return Draw.MolsToGridImage(
            mols,
            molsPerRow=len(mols),
            subImgSize=self.sub_img_size,
            legends=legends,
        )

    @staticmethod
    def _stack(rows, header):
        line_h = 18
        top = line_h * len(header) + 8
        width = max(im.width for im in rows)
        canvas = Image.new("RGB", (width, top + sum(im.height for im in rows)), "white")
        draw = ImageDraw.Draw(canvas)
        for n, line in enumerate(header):
            draw.text((8, 6 + n * line_h), line, fill="black")
        y = top
        for im in rows:
            canvas.paste(im, (0, y))
            y += im.height
        return canvas
