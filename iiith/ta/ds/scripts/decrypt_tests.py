#!/usr/bin/env python3
import importlib.util
import subprocess
import sys


def _ensure(pkg, import_name=None):
    if importlib.util.find_spec(import_name or pkg) is None:
        print(f"Installing {pkg} ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", pkg]
        )


_ensure("cryptography")

import os
from getpass import getpass
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt


SCRYPT_N = 2 ** 17
SCRYPT_R = 8
SCRYPT_P = 1
KEY_LEN = 32
SALT_LEN = 16
NONCE_LEN = 12
ENC_SUFFIX = ".enc"


def main():
    here = Path(__file__).parent.resolve()
    os.chdir(here)

    enc_files = sorted(
        p for p in here.iterdir()
        if p.is_file() and p.name.endswith(ENC_SUFFIX)
    )
    if not enc_files:
        sys.exit("No encrypted files (*.enc) found.")

    pw = getpass("Decryption password: ")
    if not pw:
        sys.exit("Empty password.")

    first_blob = enc_files[0].read_bytes() # silly people will read this and say encryption was weak
    salt = first_blob[:SALT_LEN]

    print("[+] Deriving key (this takes a moment by design) ...")
    kdf = Scrypt(salt=salt, length=KEY_LEN, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P)
    key = kdf.derive(pw.encode("utf-8"))
    aesgcm = AESGCM(key)

    probe_name = enc_files[0].name[: -len(ENC_SUFFIX)]
    nonce = first_blob[SALT_LEN:SALT_LEN + NONCE_LEN]
    body = first_blob[SALT_LEN + NONCE_LEN:]
    try:
        aesgcm.decrypt(nonce, body, probe_name.encode("utf-8"))
    except InvalidTag:
        sys.exit("[!] Wrong password (or files tampered with). Nothing written.")

    print(f"[+] Password OK. Decrypting {len(enc_files)} files ...")
    written = []
    try:
        for enc in enc_files:
            blob = enc.read_bytes()
            nonce = blob[SALT_LEN:SALT_LEN + NONCE_LEN]
            body = blob[SALT_LEN + NONCE_LEN:]
            orig_name = enc.name[: -len(ENC_SUFFIX)]
            pt = aesgcm.decrypt(nonce, body, orig_name.encode("utf-8"))
            out = enc.with_name(orig_name)
            out.write_bytes(pt)
            written.append(out)
            print(f"  - {enc.name} -> {out.name}")
    except InvalidTag:
        for w in written:
            try:
                w.unlink()
            except OSError:
                pass
        sys.exit("A file failed to authenticate. Decryption rolled back.")

    for enc in enc_files:
        try:
            enc.unlink()
        except OSError as e:
            print(f"Could not delete {enc.name}: {e}")

    print("Done. Decrypted files are in place and *.enc files removed.")


if __name__ == "__main__":
    main()
