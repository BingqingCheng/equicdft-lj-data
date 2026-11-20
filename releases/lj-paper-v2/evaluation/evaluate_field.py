"""Evaluate the frozen C4 functional on the supplied native-grid regression field."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from equicdft import GridData
from equicdft.conversion import load_converted_model


ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA = "6edbebb18192ca7f6a311a88c3236d61c3c5d285c739b9f533b4dc1b6e7137ca"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", type=Path, default=ROOT / "evaluation/nvt_T1p5_general_frame120.extxyz")
    args = parser.parse_args()
    checkpoint = ROOT / "model/model.pt"
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == MODEL_SHA
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)  # Original checkpoint was trained in float32.
    model = load_converted_model(checkpoint).double().eval().requires_grad_(False)
    torch.set_default_dtype(torch.float64)
    data = GridData.from_xyz(args.field, index="0", grid_info=model.grid_info,
                            include_local_density_index=model.requires_local_density_index)[0]
    output = model(dict(data), compute_c1=True, compute_c2=False,
                   compute_ideal=False, compute_local_mu=False)
    rho = data["rho"]
    active = rho > 1e-3
    beta = 1.0 / float(data["temperature"])
    # Lambda=1 is the classical ideal-term convention used in these data.
    raw = output["c1"][active] - rho[active].log()
    reference = beta * data["V_ext"][active]
    constant = (reference - raw).mean()
    rmse = ((raw + constant - reference).square().mean()).sqrt()
    assert torch.isfinite(output["c1"]).all() and torch.isfinite(rmse)
    print(json.dumps({"checkpoint_sha256": MODEL_SHA,
        "beta_F_exc": float(output["beta_F_exc"].detach()),
        "particle_number": float(rho.detach().sum() * 0.5**3),
        "gauge_aligned_beta_V_RMSE": float(rmse.detach()),
        "note": "One native h=0.5 field; this is not the old C3 regression model."}, indent=2))


if __name__ == "__main__":
    main()
