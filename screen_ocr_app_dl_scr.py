"""
Screen OCR desktop utility (RapidOCR + PyQt6).

A lightweight, modern desktop app that captures a screen region, 
runs deep learning OCR (RapidOCR), and performs automated actions 
(Log, Telegram, Email) based on text triggers.
"""

from __future__ import annotations

import json
import os
import smtplib
import ssl
import sys
import threading
import time
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

# Email imports for attachments
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

import difflib

# Dependency Check
try:
    import numpy as np
    import requests
    import mss
    from PIL import Image, ImageChops, ImageOps, ImageStat
    from rapidocr_onnxruntime import RapidOCR
    from PyQt6 import QtCore, QtGui, QtWidgets
except ImportError as exc:
    print("Missing requirements! Run:")
    print("pip install rapidocr_onnxruntime mss pillow requests PyQt6 numpy")
    raise SystemExit(f"Missing dependency: {exc}")

# Constants
APP_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(APP_DIR, "config.json")

def default_config() -> Dict:
    """Return default configuration values."""
    return {
        "frequency": 2.0,
        "region": {"left": 0, "top": 0, "width": 400, "height": 300},
        "downsample": 1.0,
        # Filters
        "text_filter_enabled": False,
        "text_to_detect": "Error",
        "text_similarity": 0.8,
        # Actions
        "log_to_file": False,
        "log_file_path": os.path.join(APP_DIR, "ocr_log.json"),
        "send_telegram": False,
        "telegram_token": "",
        "telegram_chat_id": "",
        "telegram_template": "Detected: {text}",
        "send_email": False,
        "include_ocr_text": True,  # Text Toggle
        "include_image": True,     # Image Toggle 
        "stop_on_match": False,    # NEW: Stop after match Toggle
        "email": {
            "server": "smtp.gmail.com",
            "port": 587,
            "use_ssl": False,
            "use_tls": True,
            "username": "",
            "password": "",
            "sender": "",
            "recipients": "",
            "subject": "OCR Alert",
            "body": "Found text:\n\n{text}",
        },
        # Change Detection
        "change_detection": True,
        "change_threshold": 2.0,
        "save_screenshots": False,
        "screenshot_dir": APP_DIR,
    }


class ConfigManager:
    """JSON-backed configuration handler."""
    def __init__(self, path: str = CONFIG_PATH):
        self.path = path
        self._config = default_config()
        self.load()

    @property
    def data(self) -> Dict:
        return self._config

    def load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
                # recursive update for nested keys (like email)
                for k, v in loaded.items():
                    if isinstance(v, dict) and k in self._config:
                        self._config[k].update(v)
                    else:
                        self._config[k] = v
        except Exception as e:
            print(f"Config load error: {e}")

    def save(self) -> None:
        try:
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(self._config, fh, indent=2)
        except Exception as e:
            print(f"Config save error: {e}")

    def reset(self) -> None:
        self._config = default_config()
        self.save()


# --- Logic Helpers ---

def capture_region(region: Dict[str, int]) -> Image.Image:
    """Capture screen region to PIL Image."""
    monitor = {
        "left": int(region.get("left", 0)),
        "top": int(region.get("top", 0)),
        "width": int(region.get("width", 100)),
        "height": int(region.get("height", 100)),
    }
    if monitor["width"] <= 0 or monitor["height"] <= 0:
         # Fallback to avoid crash if region not set
        monitor["width"] = 100
        monitor["height"] = 100
        
    with mss.mss() as sct:
        shot = sct.grab(monitor)
        img = Image.frombytes("RGB", shot.size, shot.rgb)
    return img  # Return RGB for RapidOCR

def preprocess_image(image: Image.Image, downsample: float) -> Image.Image:
    """Resize image."""
    if downsample != 1.0 and downsample > 0.0:
        new_size = (
            max(1, int(image.width * downsample)),
            max(1, int(image.height * downsample)),
        )
        image = image.resize(new_size, Image.Resampling.BILINEAR)
    return image

def mean_pixel_delta(img_a: Image.Image, img_b: Image.Image) -> float:
    """Calculate difference between two images."""
    # Convert to grayscale for simple diff
    a_gray = ImageOps.grayscale(img_a)
    b_gray = ImageOps.grayscale(img_b)
    
    # Ensure sizes match (handling potential rounding diffs in resize)
    if a_gray.size != b_gray.size:
        b_gray = b_gray.resize(a_gray.size)
        
    diff = ImageChops.difference(a_gray, b_gray)
    stat = ImageStat.Stat(diff)
    return float(stat.mean[0])

def best_fuzzy_ratio(text: str, needle: str) -> float:
    """Find best match ratio for needle in text."""
    needle = needle.strip().lower()
    haystack = text.lower()
    if not needle:
        return 1.0
        
    # Check exact substring first
    if needle in haystack:
        return 1.0
        
    # Check fuzzy similarity
    matcher = difflib.SequenceMatcher(None, needle, haystack)
    return matcher.ratio()


@dataclass
class ActionResult:
    matched: bool
    sent_telegram: bool
    sent_email: bool
    logged: bool
    message: str
    text: str


class OCRWorker(QtCore.QThread):
    """Background thread for OCR loop."""
    
    log_signal = QtCore.pyqtSignal(str)          # For activity log
    result_signal = QtCore.pyqtSignal(object)    # Returns ActionResult
    status_signal = QtCore.pyqtSignal(str)       # "Running", "Stopped"

    def __init__(self, config_manager: ConfigManager):
        super().__init__()
        self.cfg_mgr = config_manager
        self._stop_event = threading.Event()
        self._previous_image: Optional[Image.Image] = None
        self.ocr_engine = None

    def stop(self):
        self._stop_event.set()

    def run(self):
        self._stop_event.clear()
        self.status_signal.emit("Initializing Model...")
        
        # Initialize RapidOCR once
        try:
            # RapidOCR downloads models automatically if missing
            self.ocr_engine = RapidOCR() 
            self.log_signal.emit("RapidOCR model loaded.")
        except Exception as e:
            self.status_signal.emit("Model Error")
            self.log_signal.emit(f"Critial Error: Failed to load RapidOCR. {e}")
            return

        self.status_signal.emit("Running")

        while not self._stop_event.is_set():
            loop_start = time.time()
            cfg = self.cfg_mgr.data
            
            try:
                result = self._process_once(cfg)
                self.result_signal.emit(result)
                
                # --- NEW: STOP ON MATCH LOGIC ---
                if result.matched and cfg.get("stop_on_match", False):
                    self.log_signal.emit("Match found. Stopping monitor as requested.")
                    self.status_signal.emit("Stopped (Match Found)")
                    break # Exit the loop immediately

            except Exception as e:
                self.log_signal.emit(f"Loop Error: {e}")

            # Frequency wait
            freq = max(0.5, float(cfg.get("frequency", 2.0)))
            elapsed = time.time() - loop_start
            wait_time = max(0.0, freq - elapsed)
            self._stop_event.wait(wait_time)
            
        self.status_signal.emit("Stopped")

    def _process_once(self, cfg: Dict) -> ActionResult:
        # 1. Capture
        region = cfg.get("region", {})
        raw_img = capture_region(region)
        
        # 2. Downsample
        factor = float(cfg.get("downsample", 1.0))
        img = preprocess_image(raw_img, factor)
        
        # 3. Change Detection
        if cfg.get("change_detection", False) and self._previous_image:
            delta = mean_pixel_delta(img, self._previous_image)
            thresh = float(cfg.get("change_threshold", 2.0))
            if delta < thresh:
                self._previous_image = img
                return ActionResult(False, False, False, False, 
                                  f"No change (delta {delta:.2f})", "")
        
        self._previous_image = img

        # 4. Save Screenshot (Optional)
        if cfg.get("save_screenshots"):
            save_path = Path(cfg.get("screenshot_dir", APP_DIR)) / "latest_ocr.png"
            try:
                img.save(save_path)
            except Exception: pass

        # 5. Run RapidOCR
        # Convert PIL to Numpy for ONNX
        img_np = np.array(img) 
        result, _ = self.ocr_engine(img_np)
        
        text_content = ""
        if result:
            # result is list of [box, text, score]
            text_content = "\n".join([line[1] for line in result])
        
        # 6. Filtering
        matched = True
        sim_score = 1.0
        
        if cfg.get("text_filter_enabled"):
            target = cfg.get("text_to_detect", "")
            sim_score = best_fuzzy_ratio(text_content, target)
            matched = sim_score >= float(cfg.get("text_similarity", 0.8))

        # 7. Actions
        sent_tg = False
        sent_email = False
        logged = False
        
        # --- MESSAGE FORMATTING LOGIC ---
        if matched and cfg.get("text_filter_enabled"):
            clean_text = text_content.replace('\n', ' ').strip()
            if len(clean_text) > 40:
                clean_text = clean_text[:40] + "..."
            msg = f'Match "{cfg.get("text_to_detect")}" found: {clean_text}'
        else:
            msg = f"OCR Run. Text len: {len(text_content)}. Match: {matched} ({sim_score:.2f})"

        # If matched, we trigger actions
        if matched and text_content.strip():
            # Prepare image buffer only if actions enabled AND image toggle is ON
            img_buffer = None
            should_send_image = cfg.get("include_image", True)
            
            if (cfg.get("send_telegram") or cfg.get("send_email")) and should_send_image:
                img_buffer = io.BytesIO()
                img.save(img_buffer, format="PNG")
                img_buffer.seek(0)

            if cfg.get("send_telegram"):
                if img_buffer: img_buffer.seek(0)
                sent_tg = self._send_telegram(cfg, text_content, img_buffer)
                
            if cfg.get("send_email"):
                if img_buffer: img_buffer.seek(0)
                sent_email = self._send_email(cfg, text_content, img_buffer)
                
            if cfg.get("log_to_file"):
                logged = self._write_log(cfg, text_content, sent_tg, sent_email)
            
            if sent_tg or sent_email:
                suffix = " [Sent+Img]" if img_buffer else " [Sent]"
                msg += suffix

        return ActionResult(matched, sent_tg, sent_email, logged, msg, text_content)

    def _send_telegram(self, cfg: Dict, text: str, img_buffer: Optional[io.BytesIO] = None) -> bool:
        token = cfg.get("telegram_token")
        chat_id = cfg.get("telegram_chat_id")
        if not token or not chat_id: return False
        
        # Configurable: Should we include the text?
        if not cfg.get("include_ocr_text", True):
            text = "" # User opted to hide text in alert

        tpl = cfg.get("telegram_template", "{text}")
        final_msg = tpl.replace("{text}", text)
        
        try:
            # If we have an image, use sendPhoto, otherwise sendMessage
            if img_buffer:
                url = f"https://api.telegram.org/bot{token}/sendPhoto"
                # Telegram captions are limited to 1024 chars
                caption = final_msg[:1024] 
                files = {'photo': ('capture.png', img_buffer, 'image/png')}
                data = {'chat_id': chat_id, 'caption': caption}
                
                resp = requests.post(url, data=data, files=files, timeout=10)
            else:
                url = f"https://api.telegram.org/bot{token}/sendMessage"
                data = {"chat_id": chat_id, "text": final_msg}
                resp = requests.post(url, data=data, timeout=5)

            if resp.status_code != 200:
                self.log_signal.emit(f"Telegram Fail: {resp.status_code} {resp.text}")
            return resp.status_code == 200
        except Exception as e:
            self.log_signal.emit(f"Telegram Fail: {e}")
            return False

    def _send_email(self, cfg: Dict, text: str, img_buffer: Optional[io.BytesIO] = None) -> bool:
        ec = cfg.get("email", {})
        if not ec.get("server") or not ec.get("sender") or not ec.get("recipients"):
            return False

        # Configurable: Should we include the text?
        if not cfg.get("include_ocr_text", True):
            text = ""

        body_tpl = ec.get("body", "{text}")
        content = body_tpl.replace("{text}", text)
        
        # Create Multipart message
        msg = MIMEMultipart()
        msg['Subject'] = ec.get('subject', 'OCR Alert')
        msg['From'] = ec['sender']
        msg['To'] = ec['recipients'] # Should be comma sep string in header, list in sendmail

        # Attach Text
        msg.attach(MIMEText(content, 'plain'))

        # Attach Image if available
        if img_buffer:
            try:
                # Read bytes from buffer
                img_data = img_buffer.read()
                image_attachment = MIMEImage(img_data, name="capture.png")
                image_attachment.add_header('Content-Disposition', 'attachment', filename="capture.png")
                msg.attach(image_attachment)
            except Exception as e:
                self.log_signal.emit(f"Email Img Attach Fail: {e}")

        try:
            context = ssl.create_default_context() if ec.get("use_tls") or ec.get("use_ssl") else None
            
            if ec.get("use_ssl"):
                server = smtplib.SMTP_SSL(ec["server"], ec["port"], context=context, timeout=10)
            else:
                server = smtplib.SMTP(ec["server"], ec["port"], timeout=10)
                if ec.get("use_tls"):
                    server.starttls(context=context)
            
            if ec.get("username") and ec.get("password"):
                server.login(ec["username"], ec["password"])
                
            # recipients must be a list for sendmail
            recipients_list = [r.strip() for r in ec["recipients"].split(",") if r.strip()]
            server.sendmail(ec["sender"], recipients_list, msg.as_string())
            server.quit()
            return True
        except Exception as e:
            self.log_signal.emit(f"Email Fail: {e}")
            return False

    def _write_log(self, cfg: Dict, text: str, tg: bool, mail: bool) -> bool:
        path = cfg.get("log_file_path")
        if not path: return False
        
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "text": text,
            "actions": {"telegram": tg, "email": mail}
        }
        
        try:
            data = []
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except: pass
            
            data.append(entry)
            data = data[-1000:] 
            
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            self.log_signal.emit(f"Log Write Fail: {e}")
            return False


# --- GUI Components ---

class RegionSelector(QtWidgets.QWidget):
    """Transparent overlay for selecting screen region."""
    region_selected = QtCore.pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        self.start_pos = None
        self.end_pos = None
        
        # Cover all screens
        full_rect = QtCore.QRect()
        for screen in QtWidgets.QApplication.screens():
            full_rect = full_rect.united(screen.geometry())
        self.setGeometry(full_rect)

    def mousePressEvent(self, event):
        self.start_pos = event.pos()

    def mouseMoveEvent(self, event):
        self.end_pos = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        self.end_pos = event.pos()
        self.close()
        
        if not self.start_pos or not self.end_pos: return
        
        x1 = min(self.start_pos.x(), self.end_pos.x())
        y1 = min(self.start_pos.y(), self.end_pos.y())
        w = abs(self.start_pos.x() - self.end_pos.x())
        h = abs(self.start_pos.y() - self.end_pos.y())
        
        if w > 10 and h > 10:
            global_pos = self.mapToGlobal(QtCore.QPoint(x1, y1))
            self.region_selected.emit({
                "left": global_pos.x(), "top": global_pos.y(), 
                "width": w, "height": h
            })

    def paintEvent(self, event):
        qp = QtGui.QPainter(self)
        qp.setBrush(QtGui.QColor(0, 0, 0, 100))
        qp.setPen(QtCore.Qt.PenStyle.NoPen)
        qp.drawRect(self.rect())

        if self.start_pos and self.end_pos:
            rect = QtCore.QRect(self.start_pos, self.end_pos).normalized()
            qp.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_Clear)
            qp.drawRect(rect)
            
            qp.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_SourceOver)
            qp.setPen(QtGui.QPen(QtGui.QColor("red"), 2))
            qp.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            qp.drawRect(rect)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = ConfigManager()
        self.worker = OCRWorker(self.config)
        self.worker.result_signal.connect(self.on_worker_result)
        self.worker.status_signal.connect(self.update_status)
        self.worker.log_signal.connect(self.append_log)
        
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Screen OCR Monitor")
        self.resize(350, 500)
        self.setWindowFlags(QtCore.Qt.WindowType.WindowStaysOnTopHint) # Optional: keep on top

        # Central Widget
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)

        # 1. Status Area
        self.lbl_status = QtWidgets.QLabel("Status: Stopped")
        self.lbl_status.setStyleSheet("font-weight: bold; color: #aaa;")
        layout.addWidget(self.lbl_status)

        self.lbl_stats = QtWidgets.QLabel("Last Capture: None")
        self.lbl_stats.setStyleSheet("font-size: 10px; color: #888;")
        layout.addWidget(self.lbl_stats)

        # 2. Result Preview
        layout.addWidget(QtWidgets.QLabel("Latest Recognized Text:"))
        self.txt_preview = QtWidgets.QTextEdit()
        self.txt_preview.setReadOnly(True)
        # UPDATED STYLE: Dark background initially
        self.txt_preview.setStyleSheet("background: #252525; color: #ffffff; border: 1px solid #444;")
        layout.addWidget(self.txt_preview)
        
        # 3. Mini Log
        layout.addWidget(QtWidgets.QLabel("Activity Log:"))
        self.list_log = QtWidgets.QListWidget()
        self.list_log.setMaximumHeight(80)
        self.list_log.setStyleSheet("font-size: 10px; background: #252525; color: #ccc;")
        layout.addWidget(self.list_log)

        # 4. Controls
        btn_layout = QtWidgets.QHBoxLayout()
        
        self.btn_start = QtWidgets.QPushButton("Start Monitor")
        self.btn_start.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 5px;")
        self.btn_start.clicked.connect(self.toggle_worker)
        
        self.btn_region = QtWidgets.QPushButton("Select Region")
        self.btn_region.setStyleSheet("padding: 5px;")
        self.btn_region.clicked.connect(self.select_region)
        
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_region)
        layout.addLayout(btn_layout)
        
        # Menu Bar
        menu = self.menuBar()
        tools_menu = menu.addMenu("Tools")
        
        act_filter = QtGui.QAction("Filters & OCR Settings...", self)
        act_filter.triggered.connect(self.show_filter_settings)
        tools_menu.addAction(act_filter)
        
        act_actions = QtGui.QAction("Actions (Telegram/Email)...", self)
        act_actions.triggered.connect(self.show_action_settings)
        tools_menu.addAction(act_actions)
        
        act_reset = QtGui.QAction("Reset Config", self)
        act_reset.triggered.connect(self.reset_config)
        menu.addMenu("File").addAction(act_reset)

        self.update_region_label()

    # --- Actions ---

    def toggle_worker(self):
        if self.worker.isRunning():
            self.worker.stop()
            self.btn_start.setText("Start Monitor")
            self.btn_start.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 5px;")
        else:
            self.worker.start()
            self.btn_start.setText("Stop Monitor")
            self.btn_start.setStyleSheet("background-color: #c62828; color: white; font-weight: bold; padding: 5px;")

    def update_status(self, text):
        self.lbl_status.setText(f"Status: {text}")

    def on_worker_result(self, result: ActionResult):
        ts = time.strftime("%H:%M:%S")
        self.lbl_stats.setText(f"Last Capture: {ts}")
        
        if result.matched and result.text.strip():
            self.txt_preview.setText(result.text)
            # UPDATED STYLE: Dark green tint on match
            self.txt_preview.setStyleSheet("background: #1b5e20; color: #ffffff; border: 1px solid #4CAF50;") 
        else:
            # UPDATED STYLE: Revert to standard dark if no match
            self.txt_preview.setStyleSheet("background: #252525; color: #ffffff; border: 1px solid #444;")
            
        if result.message:
            self.append_log(result.message)
            
        # Update button state if stopped automatically
        if not self.worker.isRunning() and "Stopped" in self.lbl_status.text():
            self.btn_start.setText("Start Monitor")
            self.btn_start.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 5px;")

    def append_log(self, text):
        item = QtWidgets.QListWidgetItem(f"[{time.strftime('%H:%M:%S')}] {text}")
        self.list_log.addItem(item)
        self.list_log.scrollToBottom()

    def select_region(self):
        self.selector = RegionSelector()
        self.selector.region_selected.connect(self.apply_region)
        self.selector.show()

    def apply_region(self, rect):
        self.config.data["region"] = rect
        self.config.save()
        self.update_region_label()
        self.append_log(f"Region set: {rect}")

    def update_region_label(self):
        r = self.config.data["region"]
        self.btn_region.setToolTip(f"Current: {r}")

    def reset_config(self):
        self.config.reset()
        self.append_log("Config reset to defaults.")

    # --- Dialogs ---

    def show_filter_settings(self):
        d = QtWidgets.QDialog(self)
        d.setWindowTitle("Filters & OCR")
        layout = QtWidgets.QFormLayout(d)
        cfg = self.config.data
        
        # Inputs
        spin_freq = QtWidgets.QDoubleSpinBox()
        spin_freq.setRange(0.5, 60.0)
        spin_freq.setValue(float(cfg["frequency"]))
        
        spin_down = QtWidgets.QDoubleSpinBox()
        spin_down.setRange(0.1, 1.0)
        spin_down.setValue(float(cfg["downsample"]))
        
        chk_change = QtWidgets.QCheckBox("Change Detection")
        chk_change.setChecked(cfg["change_detection"])
        
        chk_filter = QtWidgets.QCheckBox("Enable Keyword Filter")
        chk_filter.setChecked(cfg["text_filter_enabled"])
        
        txt_keyword = QtWidgets.QLineEdit(cfg["text_to_detect"])
        
        spin_sim = QtWidgets.QDoubleSpinBox()
        spin_sim.setRange(0.1, 1.0)
        spin_sim.setValue(float(cfg["text_similarity"]))

        layout.addRow("Frequency (sec):", spin_freq)
        layout.addRow("Downsample:", spin_down)
        layout.addRow(chk_change)
        layout.addRow(QtWidgets.QLabel("--- Filtering ---"))
        layout.addRow(chk_filter)
        layout.addRow("Keyword:", txt_keyword)
        layout.addRow("Similarity (0-1):", spin_sim)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Save | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(d.accept)
        btns.rejected.connect(d.reject)
        layout.addRow(btns)

        if d.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            cfg["frequency"] = spin_freq.value()
            cfg["downsample"] = spin_down.value()
            cfg["change_detection"] = chk_change.isChecked()
            cfg["text_filter_enabled"] = chk_filter.isChecked()
            cfg["text_to_detect"] = txt_keyword.text()
            cfg["text_similarity"] = spin_sim.value()
            self.config.save()
            self.append_log("Filter settings saved.")

    def show_action_settings(self):
        d = QtWidgets.QDialog(self)
        d.setWindowTitle("Actions")
        d.resize(400, 500)
        layout = QtWidgets.QVBoxLayout(d)
        
        scroll = QtWidgets.QScrollArea()
        widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(widget)
        cfg = self.config.data
        
        # Logging
        chk_log = QtWidgets.QCheckBox("Log to File")
        chk_log.setChecked(cfg["log_to_file"])
        
        # Config Toggles
        chk_include_text = QtWidgets.QCheckBox("Include OCR Text in Alerts")
        chk_include_text.setChecked(cfg.get("include_ocr_text", True))
        
        chk_include_img = QtWidgets.QCheckBox("Include Screenshot in Alerts")
        chk_include_img.setChecked(cfg.get("include_image", True))
        
        chk_stop = QtWidgets.QCheckBox("Stop after first match")
        chk_stop.setChecked(cfg.get("stop_on_match", False))

        # Telegram
        chk_tg = QtWidgets.QCheckBox("Send Telegram")
        chk_tg.setChecked(cfg["send_telegram"])
        txt_tg_token = QtWidgets.QLineEdit(cfg["telegram_token"])
        txt_tg_chat = QtWidgets.QLineEdit(cfg["telegram_chat_id"])
        
        # --- NEW TEST BUTTON ---
        btn_test_tg = QtWidgets.QPushButton("Test Telegram Connection")
        btn_test_tg.clicked.connect(lambda: self.test_telegram_gui(txt_tg_token.text(), txt_tg_chat.text()))
        
        # Email
        chk_email = QtWidgets.QCheckBox("Send Email")
        chk_email.setChecked(cfg["send_email"])
        ec = cfg["email"]
        txt_srv = QtWidgets.QLineEdit(ec["server"])
        spin_port = QtWidgets.QSpinBox()
        spin_port.setRange(1, 65535)
        spin_port.setValue(ec["port"])
        txt_user = QtWidgets.QLineEdit(ec["username"])
        txt_pass = QtWidgets.QLineEdit(ec["password"])
        txt_pass.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        txt_send = QtWidgets.QLineEdit(ec["sender"])
        txt_recv = QtWidgets.QLineEdit(ec["recipients"])

        form.addRow(QtWidgets.QLabel("<b>General</b>"))
        form.addRow(chk_log)
        form.addRow(chk_include_text)
        form.addRow(chk_include_img)
        form.addRow(chk_stop) # ADDED HERE
        
        form.addRow(QtWidgets.QLabel("<b>Telegram</b>"))
        form.addRow(chk_tg)
        form.addRow("Bot Token:", txt_tg_token)
        form.addRow("Chat ID:", txt_tg_chat)
        form.addRow("", btn_test_tg) # Add button to layout
        
        form.addRow(QtWidgets.QLabel("<b>Email (SMTP)</b>"))
        form.addRow(chk_email)
        form.addRow("Server:", txt_srv)
        form.addRow("Port:", spin_port)
        form.addRow("Username:", txt_user)
        form.addRow("Password:", txt_pass)
        form.addRow("Sender Addr:", txt_send)
        form.addRow("Recipients:", txt_recv)

        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Save | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(d.accept)
        btns.rejected.connect(d.reject)
        layout.addWidget(btns)

        if d.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            cfg["log_to_file"] = chk_log.isChecked()
            cfg["include_ocr_text"] = chk_include_text.isChecked() 
            cfg["include_image"] = chk_include_img.isChecked() 
            cfg["stop_on_match"] = chk_stop.isChecked() # SAVE CONFIG
            cfg["send_telegram"] = chk_tg.isChecked()
            cfg["telegram_token"] = txt_tg_token.text()
            cfg["telegram_chat_id"] = txt_tg_chat.text()
            cfg["send_email"] = chk_email.isChecked()
            
            ec["server"] = txt_srv.text()
            ec["port"] = spin_port.value()
            ec["username"] = txt_user.text()
            ec["password"] = txt_pass.text()
            ec["sender"] = txt_send.text()
            ec["recipients"] = txt_recv.text()
            
            self.config.save()
            self.append_log("Action settings saved.")

    def test_telegram_gui(self, token, chat_id):
        """Helper to test connection directly from the dialog."""
        if not token or not chat_id:
            QtWidgets.QMessageBox.warning(self, "Error", "Please fill in Token and Chat ID first.")
            return
        
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            resp = requests.post(url, data={"chat_id": chat_id, "text": "✅ Test from OCR App!"}, timeout=5)
            
            if resp.status_code == 200:
                QtWidgets.QMessageBox.information(self, "Success", "Test message sent! Check your Telegram.")
            else:
                QtWidgets.QMessageBox.critical(self, "Failed", f"Telegram API Error:\nCode: {resp.status_code}\nMsg: {resp.text}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Connection failed:\n{e}")

    def closeEvent(self, event):
        self.worker.stop()
        self.worker.wait()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    
    # Simple Dark Theme for Modern Look
    app.setStyle("Fusion")
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(255, 255, 255))
    palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(25, 25, 25))
    palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor(255, 255, 255))
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtGui.QColor(255, 255, 255))
    palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(255, 255, 255))
    palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(255, 255, 255))
    palette.setColor(QtGui.QPalette.ColorRole.BrightText, QtGui.QColor(255, 0, 0))
    palette.setColor(QtGui.QPalette.ColorRole.Link, QtGui.QColor(42, 130, 218))
    palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(42, 130, 218))
    palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor(0, 0, 0))
    app.setPalette(palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())