#!/usr/bin/env python3
"""Offline synthetic-GT fit verification.

Generate a Pacejka data set with a chosen ground truth, run fit_pipeline on it,
and print recovery error. Used to verify solver wiring without the car.

Exit codes:
  0 — all recovered params within --tol of GT
  2 — any param outside tolerance
"""
from __future__ import annotations

import argparse
import os
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tol", type=float, default=0.10,
                        help="relative tolerance for B,C,D,E recovery (default 0.10)")
    parser.add_argument("--Bf", type=float, default=6.0)
    parser.add_argument("--Cf", type=float, default=1.9)
    parser.add_argument("--Df", type=float, default=1.0)
    parser.add_argument("--Ef", type=float, default=0.3)
    parser.add_argument("--Br", type=float, default=8.0)
    parser.add_argument("--Cr", type=float, default=1.6)
    parser.add_argument("--Dr", type=float, default=1.1)
    parser.add_argument("--Er", type=float, default=0.1)
    parser.add_argument("--noise", type=float, default=0.01)
    args = parser.parse_args(argv)

    # Make package importable when running from source tree.
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.abspath(os.path.join(here, "..", "src"))
    if src not in sys.path:
        sys.path.insert(0, src)

    from race_day_id import fit_pipeline, synthetic_gt

    model = dict(m=3.54, l_f=0.162, l_r=0.145, l_wb=0.307, h_cg=0.014,
                 C_Pf_model=[args.Bf, args.Cf, args.Df, args.Ef],
                 C_Pr_model=[args.Br, args.Cr, args.Dr, args.Er])
    gt_f = [args.Bf, args.Cf, args.Df, args.Ef]
    gt_r = [args.Br, args.Cr, args.Dr, args.Er]

    arrays = synthetic_gt.generate_dataset(model, gt_f, gt_r,
                                           noise_frac=args.noise)
    C_Pf, C_Pr, _ = fit_pipeline.fit_pacejka(arrays, model, apply_filter=False)

    def report(name, est, gt):
        print(f"{name}: est={est}  gt={gt}")
        ok = True
        for e, g, key in zip(est, gt, ("B", "C", "D", "E")):
            err = abs(e - g) / max(abs(g), 1e-6)
            marker = "OK " if err <= args.tol else "BAD"
            print(f"  {key}: err={err:.3f}  [{marker}]")
            ok = ok and err <= args.tol
        return ok

    ok_f = report("C_Pf", C_Pf, gt_f)
    ok_r = report("C_Pr", C_Pr, gt_r)
    return 0 if (ok_f and ok_r) else 2


if __name__ == "__main__":
    sys.exit(main())
