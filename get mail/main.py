import imaplib
import email
from email.header import decode_header
import os
import re
import urllib.request
import urllib.parse
from pathlib import Path
import hashlib
from datetime import datetime

from dotenv import load_dotenv

# Ladda .env i samma mapp som scriptet (eller ange sökväg)
load_dotenv()

IMAP_HOST = os.getenv("IMAP_HOST", "")
IMAP_PORT = int(os.getenv("IMAP_PORT", "993"))
IMAP_USER = os.getenv("IMAP_USER", "liam@suorsa.se")
IMAP_PASS = os.getenv("IMAP_PASS", "")
SENDER = os.getenv("SENDER", "noreply@solarweb.com")

SAVE_DIR = Path(os.getenv("SAVE_DIR", "./solarweb_downloads")).resolve()
SINCE_DATE = os.getenv("SINCE_DATE", "")          # ex: 01-Jan-2026
AUTH_BEARER = os.getenv("AUTH_BEARER", "")        # valfritt
URL_DOMAIN_HINT = os.getenv("URL_DOMAIN_HINT", "solarweb.com")


def decode_mime_words(s):
    if not s:
        return ""
    parts = decode_header(s)
    out = ""
    for part, enc in parts:
        if isinstance(part, bytes):
            out += part.decode(enc or "utf-8", errors="replace")
        else:
            out += part
    return out


def safe_filename(name: str) -> str:
    name = (name or "").strip().replace("\x00", "")
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    name = re.sub(r"\s+", " ", name)
    return name[:200] if len(name) > 200 else name


def connect_imap():
    if not IMAP_HOST:
        raise SystemExit("IMAP_HOST saknas i .env")
    if not IMAP_PASS:
        raise SystemExit("IMAP_PASS saknas i .env")

    mail = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
    mail.login(IMAP_USER, IMAP_PASS)
    return mail


def search_message_ids(mail):
    mail.select("INBOX")
    if SINCE_DATE:
        criteria = f'(FROM "{SENDER}" SINCE "{SINCE_DATE}")'
    else:
        criteria = f'(FROM "{SENDER}")'

    status, data = mail.search(None, criteria)
    if status != "OK":
        raise RuntimeError(f"Search misslyckades: {status} {data}")
    return data[0].split()


def extract_text_and_html(msg):
    text_plain = []
    text_html = []
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = (part.get("Content-Disposition") or "").lower()
            if "attachment" in disp:
                continue
            payload = part.get_payload(decode=True)
            if not payload:
                continue
            charset = part.get_content_charset() or "utf-8"
            try:
                s = payload.decode(charset, errors="replace")
            except Exception:
                s = payload.decode("utf-8", errors="replace")
            if ctype == "text/plain":
                text_plain.append(s)
            elif ctype == "text/html":
                text_html.append(s)
    else:
        payload = msg.get_payload(decode=True) or b""
        charset = msg.get_content_charset() or "utf-8"
        try:
            s = payload.decode(charset, errors="replace")
        except Exception:
            s = payload.decode("utf-8", errors="replace")
        if msg.get_content_type() == "text/html":
            text_html.append(s)
        else:
            text_plain.append(s)

    return "\n".join(text_plain), "\n".join(text_html)


def extract_urls(text_plain, text_html):
    blob = (text_plain or "") + "\n" + (text_html or "")
    urls = re.findall(r'https?://[^\s"\'<>]+', blob)

    cleaned = []
    for u in urls:
        u = u.rstrip(").,;!]")
        cleaned.append(u)

    if URL_DOMAIN_HINT:
        cleaned = [u for u in cleaned if URL_DOMAIN_HINT.lower() in u.lower()]

    seen = set()
    uniq = []
    for u in cleaned:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq


def filename_from_headers(resp, fallback_name):
    cd = resp.headers.get("Content-Disposition", "")
    m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', cd, flags=re.IGNORECASE)
    if m:
        return safe_filename(urllib.parse.unquote(m.group(1)))
    return safe_filename(fallback_name)


def download_url(url, save_dir: Path):
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Mozilla/5.0 (compatible; SolarwebDownloader/1.0)")

    if AUTH_BEARER:
        token = AUTH_BEARER.strip()
        if not token.lower().startswith("bearer "):
            token = "Bearer " + token
        req.add_header("Authorization", token)

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            final_url = resp.geturl()
            path_name = Path(urllib.parse.urlparse(final_url).path).name
            fallback = path_name if path_name else f"download_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            filename = filename_from_headers(resp, fallback)

            if "." not in filename:
                filename += ".bin"

            save_dir.mkdir(parents=True, exist_ok=True)

            data = resp.read()
            new_hash = hashlib.sha256(data).hexdigest()[:12]

            out_path = save_dir / filename
            if out_path.exists():
                old_hash = hashlib.sha256(out_path.read_bytes()).hexdigest()[:12]
                if old_hash == new_hash:
                    return None
                out_path = save_dir / f"{out_path.stem}_{new_hash}{out_path.suffix}"

            out_path.write_bytes(data)
            return out_path
    except Exception as e:
        print(f"  - Kunde inte ladda ner {url}\n    Fel: {e}")
        return None


def main():
    print(f"IMAP_USER: {IMAP_USER}")
    print(f"IMAP_HOST: {IMAP_HOST}:{IMAP_PORT}")
    print(f"Söker mail från: {SENDER}")
    if SINCE_DATE:
        print(f"SINCE: {SINCE_DATE}")
    print(f"Domänfilter: {URL_DOMAIN_HINT}")
    print(f"Sparar allt i: {SAVE_DIR}\n")

    mail = connect_imap()
    try:
        ids = search_message_ids(mail)
        print(f"Hittade {len(ids)} mail.\n")

        total_links = 0
        total_downloaded = 0

        for i, msgid in enumerate(ids, start=1):
            status, data = mail.fetch(msgid, "(RFC822)")
            if status != "OK":
                print(f"[{i}/{len(ids)}] Kunde inte hämta mail.")
                continue

            raw = data[0][1]
            msg = email.message_from_bytes(raw)
            subject = decode_mime_words(msg.get("Subject", ""))

            text_plain, text_html = extract_text_and_html(msg)
            urls = extract_urls(text_plain, text_html)

            print(f"[{i}/{len(ids)}] Ämne: {subject!r} | Länkar: {len(urls)}")
            total_links += len(urls)

            for u in urls:
                saved = download_url(u, SAVE_DIR)
                if saved:
                    total_downloaded += 1
                    print(f"  + Sparade: {saved.name}")

        print("\nKlart!")
        print(f"Totalt länkar hittade: {total_links}")
        print(f"Totalt nya filer sparade: {total_downloaded}")
        print(f"Mapp: {SAVE_DIR}")
    finally:
        try:
            mail.logout()
        except Exception:
            pass


if __name__ == "__main__":
    main()
