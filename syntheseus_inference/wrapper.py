"""Syntheseus wrapper for graphretro.

graphretro provides a real Python API (`seq_graph_retro.search.BeamSearch`),
so this wrapper loads both stage models once at init and calls
`beam_model.run_search(product, max_steps=6, rxn_class=...)` per product
in-process. Mirrors the construction in
`scripts/eval/single_edit_lg.py:104-127` but skips the accuracy bookkeeping
and exposes the results as a `List[Sequence[SingleProductReaction]]`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence

from rdkit import Chem, RDLogger
from syntheseus.interface.models import InputType, ReactionType
from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import SingleProductReaction
from syntheseus.reaction_prediction.inference.base import ExternalBackwardReactionModel
from syntheseus.reaction_prediction.utils.inference import (
    process_raw_smiles_outputs_backwards,
)

RDLogger.logger().setLevel(4)

REPO_ROOT = Path(__file__).resolve().parents[1]


class _Args:
    """Lightweight namespace mimicking argparse output for graphretro loaders."""

    def __init__(
        self,
        exp_dir: str,
        edits_exp: str,
        edits_step: str,
        lg_exp: str,
        lg_step: str,
    ) -> None:
        self.exp_dir = exp_dir
        self.edits_exp = edits_exp
        self.edits_step = edits_step
        self.lg_exp = lg_exp
        self.lg_step = lg_step


def _canonicalize_prod(p: str) -> str:
    """Canonicalize and atom-map the product, mirroring single_edit_lg.canonicalize_prod."""
    from seq_graph_retro.utils.edit_mol import canonicalize

    pcanon = canonicalize(p)
    pmol = Chem.MolFromSmiles(pcanon)
    if pmol is None:
        return p
    for atom in pmol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    return Chem.MolToSmiles(pmol)


class GraphRetroBackwardModel(ExternalBackwardReactionModel):
    """In-process wrapper around graphretro's two-stage edit + LG beam search.

    Expected `model_dir` layout (mirrors what `scripts/eval/single_edit_lg.py`
    expects when invoked with `--exp_dir <model_dir>`):

        {model_dir}/{edits_exp}/checkpoints/{edits_step}.pt
        {model_dir}/{lg_exp}/checkpoints/{lg_step}.pt

    The wrapper resolves SEQ_GRAPH_RETRO from the env or falls back to the
    repo root.
    """

    def __init__(
        self,
        *args,
        edits_exp: str = "SingleEdit_10-02-2021--08-44-37",
        edits_step: str = "epoch_156",
        lg_exp: str = "LGIndEmbed_18-02-2021--12-23-26",
        lg_step: str = "step_101951",
        beam_width: int = 10,
        max_edit_steps: int = 6,
        repo_root: Optional[Path] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.repo_root = Path(repo_root) if repo_root is not None else REPO_ROOT
        os.environ.setdefault("SEQ_GRAPH_RETRO", str(self.repo_root))
        os.environ.setdefault("WANDB_MODE", "offline")

        self.edits_exp = edits_exp
        self.edits_step = edits_step
        self.lg_exp = lg_exp
        self.lg_step = lg_step
        self.beam_width = int(beam_width)
        self.max_edit_steps = int(max_edit_steps)

        # Validate checkpoints exist before paying the import cost.
        model_dir = Path(self.model_dir)
        for exp, step in (
            (edits_exp, edits_step),
            (lg_exp, lg_step),
        ):
            cp = model_dir / exp / "checkpoints" / f"{step}.pt"
            if not cp.exists():
                raise FileNotFoundError(
                    f"graphretro checkpoint missing: {cp}. "
                    "Download from the upstream vsomnath/graphretro Drive and "
                    "stage the per-experiment dirs under your --model-dir."
                )

        # Build the two stage models + beam search wrapper, mirroring
        # scripts/eval/single_edit_lg.py:104-127.
        from scripts.eval.single_edit_lg import load_edits_model, load_lg_model
        from seq_graph_retro.models import EditLGSeparate
        from seq_graph_retro.search import BeamSearch

        loader_args = _Args(
            exp_dir=str(model_dir),
            edits_exp=edits_exp,
            edits_step=edits_step,
            lg_exp=lg_exp,
            lg_step=lg_step,
        )
        edits_loaded, edit_net_name = load_edits_model(loader_args)
        lg_loaded, lg_net_name = load_lg_model(loader_args)

        edits_config = edits_loaded["saveables"]
        lg_config = lg_loaded["saveables"]
        lg_toggles = lg_config.get("toggles", {})

        # Patch tensor_file path if it points at a missing absolute location.
        if "tensor_file" in lg_config and not os.path.isfile(lg_config["tensor_file"]):
            data_dir = self.repo_root / "datasets" / "uspto-50k"
            if lg_toggles.get("use_rxn_class", False):
                lg_config["tensor_file"] = str(
                    data_dir / "train" / "h_labels" / "with_rxn" / "lg_inputs.pt"
                )
            else:
                lg_config["tensor_file"] = str(
                    data_dir / "train" / "h_labels" / "without_rxn" / "lg_inputs.pt"
                )

        rm = EditLGSeparate(
            edits_config=edits_config,
            lg_config=lg_config,
            edit_net_name=edit_net_name,
            lg_net_name=lg_net_name,
            device=self.device,
        )
        rm.load_state_dict(edits_loaded["state"], lg_loaded["state"])
        rm.to(self.device)
        rm.eval()

        self._lg_toggles = lg_toggles
        self.beam_model = BeamSearch(
            model=rm, beam_width=self.beam_width, max_edits=1
        )

    def get_parameters(self):
        return self.beam_model.model.parameters()

    # ----- syntheseus dispatch / cache -----
    def __call__(
        self,
        inputs: list[InputType],
        num_results: Optional[int] = None,
        reaction_types=None,
    ) -> list[Sequence[ReactionType]]:
        num_results = num_results or self.default_num_results
        inputs_not_in_cache = list(
            {inp for inp in inputs if (inp, num_results) not in self._cache}
        )
        if len(inputs_not_in_cache) > 0:
            new_rxns = self._get_reactions(
                inputs=inputs_not_in_cache,
                num_results=num_results,
                reaction_types=reaction_types,
            )
            assert len(new_rxns) == len(inputs_not_in_cache)
            for inp, rxns in zip(inputs_not_in_cache, new_rxns):
                self._cache[(inp, num_results)] = self.filter_reactions(rxns)

        output = [self._cache[(inp, num_results)] for inp in inputs]
        if not self._use_cache:
            self._cache.clear()

        self._num_cache_misses += len(inputs_not_in_cache)
        self._num_cache_hits += len(inputs) - len(inputs_not_in_cache)

        return output

    # ----- core inference -----
    def _get_reactions(
        self,
        inputs: List[Molecule],
        num_results: int,
        reaction_types=None,
    ) -> List[Sequence[SingleProductReaction]]:
        from seq_graph_retro.utils.edit_mol import generate_reac_set

        use_rxn_class = bool(self._lg_toggles.get("use_rxn_class", False))

        out: List[Sequence[SingleProductReaction]] = []
        for input_idx, mol in enumerate(inputs):
            p = _canonicalize_prod(mol.smiles)
            rxn_class = None
            if use_rxn_class and reaction_types is not None:
                try:
                    rxn_class = int(reaction_types[input_idx])
                except (TypeError, IndexError, ValueError):
                    rxn_class = None

            try:
                if rxn_class is not None:
                    top_k_nodes = self.beam_model.run_search(
                        p, max_steps=self.max_edit_steps, rxn_class=rxn_class
                    )
                else:
                    top_k_nodes = self.beam_model.run_search(
                        p, max_steps=self.max_edit_steps
                    )
            except Exception as e:  # graphretro can raise on tricky molecules
                print(f"[graphretro] beam search failed for {p}: {e}", flush=True)
                top_k_nodes = []

            raw_outputs: List[str] = []
            metadata_list: List[dict] = []
            for beam_idx, node in enumerate(top_k_nodes[:num_results]):
                pred_edit = node.edit
                pred_label = node.lg_groups
                if isinstance(pred_edit, list):
                    pred_edit = pred_edit[0]
                try:
                    pred_set = generate_reac_set(p, pred_edit, pred_label, verbose=False)
                except Exception as e:
                    print(f"[graphretro] generate_reac_set failed: {e}", flush=True)
                    pred_set = None
                if not pred_set:
                    continue
                # `pred_set` is a Python set of '.'-joined reactant SMILES strings.
                for reactant_str in pred_set:
                    raw_outputs.append(reactant_str)
                    metadata_list.append(
                        {
                            "probability": getattr(node, "prob", None),
                            "rank": beam_idx,
                        }
                    )

            out.append(
                process_raw_smiles_outputs_backwards(
                    input=mol,
                    output_list=raw_outputs,
                    metadata_list=metadata_list,
                )
            )
        return out
