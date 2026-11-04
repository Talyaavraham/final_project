"""
Pull only the selected UCF-Crime videos out of the 103 GB archive, by HTTP
range requests (no full download). Selection: eval/data/ucf_selection.json
(made from the UCA annotations). Resumable: existing files are skipped.

    python eval/fetch_ucf.py            # all selected
    python eval/fetch_ucf.py --limit 5  # first five, to test
"""
import argparse
import json
import os
import sys
import time

import truststore
truststore.inject_into_ssl()           # the UCF server sends an incomplete chain; use the OS trust store
from remotezip import RemoteZip

URL = "https://www.crcv.ucf.edu/data1/chenchen/UCF_Crimes.zip"
HERE = os.path.dirname(os.path.abspath(__file__))
SEL = os.path.join(HERE, "data", "ucf_selection.json")
OUT = os.path.join(HERE, "data", "ucf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)   # the server throttles per IP; more is not faster
    args = ap.parse_args()
    sel = json.load(open(SEL))
    names = sorted(sel, key=lambda n: sel[n]["bytes"])
    if args.limit:
        names = names[:args.limit]
    os.makedirs(OUT, exist_ok=True)
    todo = [n for n in names if not os.path.exists(os.path.join(OUT, n + ".mp4"))]
    total = sum(sel[n]["bytes"] for n in todo)
    print(f"{len(todo)} of {len(names)} videos to fetch, {total/1e9:.2f} GB", flush=True)
    t0 = time.time()
    state = {"done": 0, "n": 0}

    def fetch(n):
        member = sel[n]["path"]
        dst = os.path.join(OUT, n + ".mp4")
        for attempt in range(4):
            try:
                # timeout: a stalled connection must fail and retry, not hang for hours
                with RemoteZip(URL, initial_buffer_size=512 << 10, timeout=(30, 120)) as z:
                    with z.open(member) as src, open(dst + ".part", "wb") as f:
                        while True:
                            chunk = src.read(4 << 20)
                            if not chunk:
                                break
                            f.write(chunk)
                os.replace(dst + ".part", dst)
                return True
            except Exception as e:  # noqa: BLE001
                print(f"  retry {attempt+1} for {n}: {type(e).__name__}: {str(e)[:80]}", flush=True)
                time.sleep(10 * (attempt + 1))
        print(f"FAILED {n}", flush=True)
        return False

    # the server gives ~0.7 MB/s per connection: run several members at once
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch, n): n for n in todo}
        for fut in as_completed(futs):
            n = futs[fut]
            state["n"] += 1
            if fut.result():
                state["done"] += sel[n]["bytes"]
            rate = state["done"] / max(1, time.time() - t0) / 1e6
            print(f"[{state['n']}/{len(todo)}] {n} {sel[n]['bytes']/1e6:.0f} MB  ({rate:.1f} MB/s, "
                  f"{(total-state['done'])/1e9:.2f} GB left)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
