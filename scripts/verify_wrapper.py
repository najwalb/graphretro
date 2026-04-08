"""Standalone smoke test for the graphretro syntheseus wrapper.

Run from the repo root inside the `graphretro` conda env:

    python scripts/verify_wrapper.py --model-dir models

This:
  1. Imports the wrapper (forces SEQ_GRAPH_RETRO + WANDB_MODE=offline).
  2. Boots the model — loads both stage checkpoints into memory.
  3. If --run is passed, runs one full inference call on aspirin and prints
     the top-3 reactant predictions.

The boot step requires the staged checkpoints to exist.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("REPO_ROOT", str(REPO_ROOT))
os.environ.setdefault("SEQ_GRAPH_RETRO", str(REPO_ROOT))
os.environ.setdefault("WANDB_MODE", "offline")
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Staged checkpoint dir")
    parser.add_argument(
        "--edits-exp", default="SingleEdit_10-02-2021--08-44-37"
    )
    parser.add_argument("--edits-step", default="epoch_156")
    parser.add_argument("--lg-exp", default="LGIndEmbed_18-02-2021--12-23-26")
    parser.add_argument("--lg-step", default="step_101951")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    from syntheseus.interface.molecule import Molecule

    from syntheseus_inference.wrapper import GraphRetroBackwardModel

    import torch

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[verify] device={device} model_dir={args.model_dir}")

    model = GraphRetroBackwardModel(
        model_dir=args.model_dir,
        device=device,
        default_num_results=5,
        use_cache=False,
        edits_exp=args.edits_exp,
        edits_step=args.edits_step,
        lg_exp=args.lg_exp,
        lg_step=args.lg_step,
        beam_width=10,
        max_edit_steps=6,
        repo_root=REPO_ROOT,
    )
    print("[verify] wrapper instantiated successfully")

    if args.run:
        aspirin = Molecule("CC(=O)Oc1ccccc1C(=O)O")
        results = model([aspirin], num_results=3)
        print(f"[verify] inference returned {len(results)} batches, "
              f"first batch has {len(results[0])} reactions")
        for i, rxn in enumerate(results[0]):
            reactants = ".".join(m.smiles for m in rxn.reactants)
            print(f"  rank {i}: {reactants}")


if __name__ == "__main__":
    main()
