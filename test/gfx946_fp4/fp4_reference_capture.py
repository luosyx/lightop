#!/usr/bin/env python3
"""Capture or compare paged_mqa_logits_fp4 output across extension builds.

Used by gfx946_cardtime.sh to show the native FP4 path returns the same logits
as the shipping path:

    --save REF      run the operator, write logits to REF
    --compare REF   run it again, compare against REF

The two runs come from *different builds* (native off, then native on with the
strict fallback), so this is a real differential rather than a self-comparison.
With strict mode on, any tile the native path rejects aborts the process, so a
clean compare also proves every scored tile went through the native MMAC.

Inputs keep block scales near the 127 bias so the native path's eligibility
gate is satisfied by construction; otherwise strict mode would trap on inputs
that are perfectly legal for the shipping path.
"""
import argparse
import sys

import torch

# heads, max_context_len, table_block_size
CONFIGS = [
    (16, 512, 64),
    (32, 512, 64),
    (64, 512, 64),
    (32, 1024, 64),
    (32, 512, 8),
]
BATCH = 8


def _op():
    import lightop
    for holder in (getattr(lightop, "op", None), lightop,
                   getattr(getattr(lightop, "attention", None), "mqa", None)):
        fn = getattr(holder, "paged_mqa_logits_fp4", None)
        meta = getattr(holder, "get_paged_mqa_logits_fp4_metadata", None)
        if fn is not None:
            return fn, meta
    raise RuntimeError("paged_mqa_logits_fp4 not found in the lightop package")


def build_inputs(device, *, heads, max_len, table_block_size, seed):
    g = torch.Generator().manual_seed(seed)

    # Packed E2M1, 64 bytes per head = 128 channels.
    q = torch.randint(0, 256, (BATCH, 1, heads, 64), generator=g, dtype=torch.uint8)

    # Four UE8M0 scales per head, packed into one int32, all near the bias.
    sb = torch.randint(120, 135, (BATCH, 1, heads, 4), generator=g, dtype=torch.int32)
    q_sf = (sb[..., 0] | (sb[..., 1] << 8) | (sb[..., 2] << 16) | (sb[..., 3] << 24))

    blocks = (max_len + table_block_size - 1) // table_block_size
    # Enough pages that every column of every row maps to a real pool slot.
    slots_needed = blocks * table_block_size
    pages = max(1, (slots_needed + 63) // 64)

    # Fused index-K pool page, 4352 bytes.  The kernel addresses it as two
    # contiguous regions, NOT as 68 bytes per slot:
    #     payload  page + slot * 64            bytes [0, 4096)
    #     scales   page + 64 * 64 + slot * 4   bytes [4096, 4352)
    # Filling a [pages, 64, 1, 68] view per slot instead puts the scale bytes
    # inside payload territory, and the kernel then reads random payload bytes
    # as UE8M0 -- a byte of 255 becomes 2^128 and the logits overflow to inf.
    kv = torch.empty((pages, 4352), dtype=torch.uint8)
    kv[:, :4096] = torch.randint(0, 256, (pages, 4096), generator=g, dtype=torch.uint8)
    kv[:, 4096:] = torch.randint(120, 135, (pages, 256), generator=g, dtype=torch.uint8)

    # slot = block_table[b, j // TB] * TB + j % TB  must stay inside pages*64,
    # so the entry bound depends on TB -- pages for TB=64, pages*8 for TB=8.
    entry_bound = pages * 64 // table_block_size
    block_table = torch.randint(0, entry_bound, (BATCH, blocks),
                                generator=g, dtype=torch.int32)

    weights = torch.randn((BATCH, heads), generator=g, dtype=torch.float32)
    context_lens = torch.randint(1, max_len + 1, (BATCH,), generator=g, dtype=torch.int32)

    return dict(
        q=q.to(device), q_sf=q_sf.contiguous().to(device), kv_cache=kv.to(device),
        weights=weights.to(device), context_lens=context_lens.to(device),
        block_table=block_table.contiguous().to(device),
        max_context_len=max_len, table_block_size=table_block_size,
    )


def run_all(device):
    fn, meta_fn = _op()
    cus = torch.cuda.get_device_properties(0).multi_processor_count
    results = {}

    for idx, (heads, max_len, tbs) in enumerate(CONFIGS):
        a = build_inputs(device, heads=heads, max_len=max_len,
                         table_block_size=tbs, seed=1000 + idx)
        for persistent in (False, True):
            schedule = None
            if persistent:
                if meta_fn is None:
                    continue
                try:
                    schedule = meta_fn(a["context_lens"], 64, cus)
                except Exception as exc:
                    print(f"  cfg{idx} persistent: planner declined ({exc})")
                    continue
            logits = fn(a["q"], a["q_sf"], a["kv_cache"], a["weights"],
                        a["context_lens"], a["block_table"], schedule,
                        a["max_context_len"], a["table_block_size"])
            # Columns at or past context_lens[b] are left unwritten by contract,
            # so they hold whatever the allocator left behind -- possibly NaN.
            # Select rather than multiply: NaN * 0 is still NaN, and because
            # every comparison against NaN is False, that would later make the
            # tolerance check silently report a pass.
            cols = torch.arange(logits.shape[1], device=logits.device)
            mask = cols.unsqueeze(0) < a["context_lens"].unsqueeze(1).to(cols.device)
            clean = torch.where(mask, logits, torch.zeros_like(logits))
            if not torch.isfinite(clean).all():
                raise RuntimeError(
                    f"non-finite value inside the visible region of h{heads}/"
                    f"len{max_len}/tb{tbs} -- a real operator bug, not an"
                    " unwritten column")
            key = f"h{heads}_len{max_len}_tb{tbs}_{'persist' if persistent else 'grid'}"
            results[key] = clean.float().cpu()
            print(f"  {key}: {tuple(logits.shape)} ok")

    if not results:
        raise RuntimeError("no configuration produced output")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save")
    p.add_argument("--compare")
    args = p.parse_args()
    if bool(args.save) == bool(args.compare):
        p.error("give exactly one of --save / --compare")

    if not torch.cuda.is_available():
        print("no GPU"); return 1
    device = "cuda"
    arch = torch.cuda.get_device_properties(0).gcnArchName
    print(f"device {arch}")

    got = run_all(device)

    if args.save:
        torch.save(got, args.save)
        print(f"saved {len(got)} tensors to {args.save}")
        return 0

    ref = torch.load(args.compare)
    missing = set(ref) ^ set(got)
    if missing:
        print(f"FAIL configuration sets differ: {sorted(missing)}")
        return 1

    worst, worst_key, bad = 0.0, None, 0
    for k in sorted(ref):
        a, b = ref[k], got[k]
        if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
            print(f"  {k}: NON-FINITE values present -> FAIL")
            bad += 1
            continue
        denom = a.abs().amax().clamp_min(1e-30)
        rel = ((b - a).abs().amax() / denom).item()
        if rel > worst:
            worst, worst_key = rel, k
        # Phrased as `not (rel <= tol)` so a NaN fails: `rel > tol` would be
        # False for NaN and would let a broken run report a pass.
        failed = not (rel <= 1e-5)
        if failed:
            bad += 1
        print(f"  {k}: max rel {rel:.3g} {'DIFF' if failed else 'ok'}")

    print(f"worst: {worst:.3g} at {worst_key}")
    if bad:
        print(f"FAIL {bad} configuration(s) differ beyond 1e-5")
        return 1
    print("PASS native output matches the shipping path")
    return 0


if __name__ == "__main__":
    sys.exit(main())
