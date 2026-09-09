import sys, torch
try:
    ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    ep = ck.get("epoch", "ERR")
    seed = ck.get("seed", "")
    sha = ck.get("cfg_sha", "")
    print(f"{ep} {seed} {sha}")
except Exception:
    print("ERR ERR ERR")
