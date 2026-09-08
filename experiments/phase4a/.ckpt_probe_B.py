import sys, torch
try:
    ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    print(ck.get("epoch", "ERR"))
except Exception:
    print("ERR")
