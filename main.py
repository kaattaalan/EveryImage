import sys
import os
import time
import threading
import argparse
import socket
import urllib.request
from pathlib import Path
import uvicorn

WORKSPACE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE_ROOT))
import src.config as cfg
import src.database as db

def is_server_already_running(host: str, port: int) -> bool:
    try:
        req = urllib.request.Request(f"http://{host}:{port}/api/stats")
        with urllib.request.urlopen(req, timeout=0.8) as resp:
            return resp.status == 200
    except Exception:
        return False

def find_free_port(host: str, start_port: int = 8899, max_attempts: int = 20) -> int:
    for p in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, p))
                return p
            except OSError:
                continue
    return start_port

def warmup_engines():
    try:
        from src.models import SiglipEngine, FaceEngine
        SiglipEngine.get_instance()
        FaceEngine.get_instance()
    except Exception:
        pass

def start_api_server(host="127.0.0.1", port=8899):
    from src.api import app
    threading.Thread(target=warmup_engines, daemon=True).start()
    config = uvicorn.Config(app=app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    server.run()

def run_desktop(host="127.0.0.1", port=8899):
    db.init_db()

    # Check if EveryImage is ALREADY running on the port
    if is_server_already_running(host, port):
        print(f"EveryImage server is already active at http://{host}:{port}. Attaching desktop window...")
        actual_port = port
    else:
        # Find a genuinely available port if port has socket conflict
        actual_port = find_free_port(host, port)
        server_thread = threading.Thread(target=start_api_server, args=(host, actual_port), daemon=True)
        server_thread.start()
        # Wait briefly for server to bind
        time.sleep(1.0)

    app_url = f"http://{host}:{actual_port}"
    print(f"EveryImage application ready at: {app_url}")

    # Launch PyWebView desktop window
    try:
        import webview
        print("Launching native desktop window via WebView2...")
        webview_storage = str(cfg.CACHE_TEMP / "webview")
        window = webview.create_window(
            title="EveryImage - Desktop Image Indexer",
            url=app_url,
            width=1280,
            height=850,
            min_size=(920, 600),
            background_color="#0b0f19"
        )
        webview.start(storage_path=webview_storage, debug=False)
    except Exception as e:
        print(f"Could not open native WebView window ({e}). Opening in default browser...")
        import webbrowser
        webbrowser.open(app_url)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Shutting down EveryImage...")

def main():
    parser = argparse.ArgumentParser(description="EveryImage Desktop Application")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8899, help="Port (default: 8899)")
    parser.add_argument("--server-only", action="store_true", help="Run HTTP server only without opening UI window")
    args = parser.parse_args()

    if args.server_only:
        db.init_db()
        if is_server_already_running(args.host, args.port):
            print(f"EveryImage is already running on http://{args.host}:{args.port}")
        else:
            actual_port = find_free_port(args.host, args.port)
            start_api_server(args.host, actual_port)
    else:
        run_desktop(args.host, args.port)

if __name__ == "__main__":
    main()
