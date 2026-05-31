#!/usr/bin/env python
# coding: utf-8
"""CheMeleon learned molecular fingerprint adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence
from urllib.request import urlretrieve

import numpy as np
from rdkit.Chem import Mol, MolFromSmiles

DEFAULT_CHEMELEON_CHECKPOINT_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
DEFAULT_CHEMELEON_CHECKPOINT_DIR = Path("data/model_assets/checkpoints/chemeleon").resolve()
DEFAULT_CHEMELEON_CHECKPOINT_NAME = "chemeleon_mp.pt"


class CheMeleonFingerprint:
    """Generate CheMeleon learned embeddings for SMILES strings or RDKit molecules.

    This is a local adapter around the standalone CheMeleon fingerprint recipe from
    Jackson Burns' CheMeleon repository. Imports are intentionally lazy so regular
    feature tooling remains importable without the optional Chemprop runtime.
    """

    def __init__(
        self,
        *,
        device: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        allow_auto_download: bool = True,
        checkpoint_url: str = DEFAULT_CHEMELEON_CHECKPOINT_URL,
    ) -> None:
        try:
            import torch
            from chemprop import featurizers, nn
            from chemprop.models import MPNN
            from chemprop.nn import RegressionFFN
        except Exception as exc:  # pragma: no cover - exercised via toolkit error path
            raise RuntimeError(
                "CheMeleon fingerprints require the optional Chemprop runtime. "
                "Install the `prediction` extras or `chemprop>=2.2.0`."
            ) from exc

        self._torch = torch
        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        resolved_checkpoint = self._resolve_checkpoint_path(
            checkpoint_path=checkpoint_path,
            checkpoint_dir=checkpoint_dir,
            allow_auto_download=allow_auto_download,
            checkpoint_url=checkpoint_url,
        )

        chemeleon_mp = torch.load(resolved_checkpoint, weights_only=True, map_location="cpu")
        mp = nn.BondMessagePassing(**chemeleon_mp["hyper_parameters"])
        mp.load_state_dict(chemeleon_mp["state_dict"])
        self.model = MPNN(
            message_passing=mp,
            agg=nn.MeanAggregation(),
            predictor=RegressionFFN(input_dim=mp.output_dim),
        )
        self.model.eval()
        if device is not None:
            self.model.to(device=device)

    def _resolve_checkpoint_path(
        self,
        *,
        checkpoint_path: Optional[str],
        checkpoint_dir: Optional[str],
        allow_auto_download: bool,
        checkpoint_url: str,
    ) -> Path:
        if checkpoint_path:
            path = Path(checkpoint_path).expanduser().resolve()
        else:
            root = (
                Path(checkpoint_dir).expanduser().resolve()
                if checkpoint_dir
                else DEFAULT_CHEMELEON_CHECKPOINT_DIR
            )
            path = root / DEFAULT_CHEMELEON_CHECKPOINT_NAME

        if path.exists():
            return path
        if not allow_auto_download:
            raise FileNotFoundError(
                f"CheMeleon checkpoint not found at {path}. Provide `checkpoint_path` "
                "or allow automatic download."
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(checkpoint_url, path)
        return path

    def __call__(self, molecules: Sequence[str | Mol]) -> np.ndarray:
        from chemprop.data import BatchMolGraph

        mols = []
        for molecule in molecules:
            mol = MolFromSmiles(molecule) if isinstance(molecule, str) else molecule
            if mol is None:
                raise ValueError(f"Could not featurize molecule for CheMeleon: {molecule}")
            mols.append(mol)

        bmg = BatchMolGraph([self.featurizer(mol) for mol in mols])
        device = next(self.model.parameters()).device
        bmg.to(device=device)
        with self._torch.no_grad():
            fingerprints = self.model.fingerprint(bmg)
        return fingerprints.detach().cpu().numpy()
