"""Resumable, checksum-verified download of the two public datasets (low CPU; uses curl)."""
import hashlib, json, os, subprocess, sys, time, urllib.request

RECORDS = {
    4751737: ["DeepLIIF_Training_Set.zip", "DeepLIIF_Validation_Set.zip", "DeepLIIF_Testing_Set.zip"],
    15370205: ["tissuenet_v1.1.zip"],
}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main(out):
    os.makedirs(out, exist_ok=True)
    for rec, names in RECORDS.items():
        req = urllib.request.Request(f"https://zenodo.org/api/records/{rec}", headers={"User-Agent": "research-client"})
        meta = json.loads(urllib.request.urlopen(req, timeout=60).read())
        files = {f["key"]: f for f in meta["files"]}
        for name in names:
            f, path = files[name], os.path.join(out, name)
            want = f["checksum"].split(":")[1]
            for attempt in range(40):
                have = os.path.getsize(path) if os.path.exists(path) else 0
                if have >= f["size"]:
                    break
                print(f"{name}: {have}/{f['size']} attempt {attempt}", flush=True)
                subprocess.call(["curl", "-sSL", "-C", "-", "--connect-timeout", "30", "-o", path, f["links"]["self"]])
                time.sleep(5)
            ok = os.path.getsize(path) == f["size"] and md5(path) == want
            print(f"{name}: size={os.path.getsize(path)} md5_ok={ok}", flush=True)
            if not ok:
                sys.exit(f"checksum failed: {name}")
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
