"""Minimal HTTPS downloader that caps the TCP segment size (MSS) so that
servers behind a small-MTU tunnel (e.g. plato.asu.edu through a VPN with
MTU 1300) complete the TLS handshake.

    fetch_url.py https://host/path out_file [mss]
"""
import socket
import ssl
import sys
import urllib.parse

url, out = sys.argv[1], sys.argv[2]
mss = int(sys.argv[3]) if len(sys.argv) > 3 else 1200
for hops in range(5):
    u = urllib.parse.urlparse(url)
    host, port = u.hostname, u.port or (443 if u.scheme == "https" else 80)
    path = u.path + ("?" + u.query if u.query else "")
    s = None
    for attempt in range(8):  # the path is flaky: retry with a smaller MSS each time
        try:
            raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_MAXSEG, max(536, mss - 100 * (attempt // 2)))
            raw.settimeout(25)
            raw.connect((host, port))
            s = ssl.create_default_context().wrap_socket(raw, server_hostname=host) if u.scheme == "https" else raw
            s.settimeout(120)
            break
        except (OSError, ssl.SSLError):
            raw.close()
            s = None
    if s is None:
        sys.exit("could not connect")
    s.sendall(f"GET {path} HTTP/1.1\r\nHost: {host}\r\nUser-Agent: fetch/1\r\nConnection: close\r\n\r\n".encode())
    buf = b""
    while b"\r\n\r\n" not in buf:
        d = s.recv(65536)
        if not d:
            break
        buf += d
    head, _, body = buf.partition(b"\r\n\r\n")
    status = int(head.split()[1])
    hdr = {l.split(b":", 1)[0].lower().decode(): l.split(b":", 1)[1].strip().decode()
           for l in head.split(b"\r\n")[1:] if b":" in l}
    if status in (301, 302, 307, 308):
        url = urllib.parse.urljoin(url, hdr["location"])
        s.close()
        continue
    if status != 200:
        sys.exit(f"HTTP {status}")
    total = int(hdr.get("content-length", 0))
    got = len(body)
    with open(out, "wb") as f:
        f.write(body)
        while True:
            d = s.recv(1 << 20)
            if not d:
                break
            f.write(d)
            got += len(d)
    print(f"{out}: {got} bytes" + (f" of {total}" if total else ""))
    break
