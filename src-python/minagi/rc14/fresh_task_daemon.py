from __future__ import annotations

import argparse
import os
from pathlib import Path
import secrets
import signal
import stat

from .fresh_tasks import DurableFreshTaskAuthority
from .fresh_task_service import FreshTaskAuthorityServer


def _load_or_create_token(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink():
            raise RuntimeError(f"fresh-task capability token must not be a symlink: {path}")
        st = path.stat()
        if not stat.S_ISREG(st.st_mode):
            raise RuntimeError(f"fresh-task capability token must be a regular file: {path}")
        if stat.S_IMODE(st.st_mode) & 0o077:
            raise RuntimeError(f"fresh-task capability token permissions must not grant group/other access: {path}")
        token = path.read_text(encoding="utf-8").strip()
        if len(token) < 32:
            raise RuntimeError(f"fresh-task capability token too short: {path}")
        return token
    token = secrets.token_hex(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, (token + "\n").encode("utf-8"))
    finally:
        os.close(fd)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return token


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run the isolated authenticated RC14 fresh-task authority service")
    p.add_argument("--root", required=True, help="private authority state directory")
    p.add_argument("--socket", required=True, help="Unix-domain socket exposed to evaluator clients")
    p.add_argument("--key-path", help="optional AES-256 key path; place outside research-readable storage")
    p.add_argument("--evaluator-token-file", help="evaluator capability token file; default under private root")
    p.add_argument("--admin-token-file", help="admin capability token file; default under private root")
    args = p.parse_args(argv)
    root = Path(args.root)
    evaluator_token_path = Path(args.evaluator_token_file) if args.evaluator_token_file else root / ".evaluator.cap"
    admin_token_path = Path(args.admin_token_file) if args.admin_token_file else root / ".admin.cap"
    evaluator_token = _load_or_create_token(evaluator_token_path)
    admin_token = _load_or_create_token(admin_token_path)
    if secrets.compare_digest(evaluator_token, admin_token):
        raise RuntimeError("evaluator/admin capability tokens must differ")
    authority = DurableFreshTaskAuthority(root, key_path=args.key_path)
    server = FreshTaskAuthorityServer(Path(args.socket), authority, evaluator_token=evaluator_token, admin_token=admin_token)

    def stop(*_):
        server.stop()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
