from pathlib import Path

scanner_path = Path("src/scanner.py")
content = scanner_path.read_text(encoding="utf-8")

# 1. Update crawling loop to report discovered files dynamically
old_crawl = '''            for lib in libs_to_scan:
                lib_path = Path(lib["path"])
                if not lib_path.exists():
                    continue
                for root, _, files in os.walk(lib_path):
                    if self.stop_requested.is_set():
                        break
                    for f in files:
                        p = Path(root) / f
                        if p.suffix.lower() in SUPPORTED_EXTS:
                            file_queue.append((lib["id"], p))'''

new_crawl = '''            for lib in libs_to_scan:
                lib_path = Path(lib["path"])
                if not lib_path.exists():
                    continue
                for root, _, files in os.walk(lib_path):
                    if self.stop_requested.is_set():
                        break
                    for f in files:
                        p = Path(root) / f
                        if p.suffix.lower() in SUPPORTED_EXTS:
                            file_queue.append((lib["id"], p))
                            if len(file_queue) % 10 == 0:
                                with self.lock:
                                    self.total_files = len(file_queue)
                                    self.current_file = f"Discovered {len(file_queue)} photos in {lib_path.name}..."'''

content = content.replace(old_crawl, new_crawl)

# 2. Change batch_size from 8 to 4 for faster UI feedback
content = content.replace("batch_size = 8", "batch_size = 4")

# 3. Inside _process_batch, update self.current_file and self.speed_fps per image
old_loop = '''            with self.lock:
                self.tagged_files += 1'''

new_loop = '''            with self.lock:
                self.current_file = fpath.name
                self.tagged_files += 1
                elapsed = max(0.1, time.time() - self.start_time)
                self.speed_fps = self.tagged_files / elapsed'''

content = content.replace(old_loop, new_loop)

scanner_path.write_text(content, encoding="utf-8")
print("Updated src/scanner.py for dynamic progress bar successfully!")
