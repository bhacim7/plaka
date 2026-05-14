import os

# Bellek parçalanmasını engelleyen koruma kalkanı
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
os.environ["OMP_NUM_THREADS"] = "1"

import torch

# İŞTE EKSİK OLAN VE HATAYI ÇÖZECEK KRİTİK SATIR BURADA!
# PyTorch'un sorunlu cuDNN motorunu tamamen kapatıp standart CUDA çekirdeklerini kullanmaya zorluyoruz.
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.set_num_threads(1)

import cv2
cv2.setNumThreads(0)
from ultralytics import YOLO
import easyocr
import numpy as np
import re
from itertools import product
import time
import threading
import queue
import socket
import struct
import pickle

# ─────────────────────────────────────────────
# JETSON NANO KAMERA VE YAYIN AYARLARI
# ─────────────────────────────────────────────
CAMERA_MODE = 'usb'
USB_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS_TARGET = 30

YAYIN_YAP = True
SUNUCU_IP = "192.168.1.34"
SUNUCU_PORT = 5000
STREAM_WIDTH = 640
STREAM_HEIGHT = 360
JPEG_KALITE = 45

YOLO_MODEL_YOLU = 'weights (1).engine'

# GPU TRAFİK POLİSİ (Çarpışmaları Önler)
gpu_lock = threading.Lock()

# ─────────────────────────────────────────────
# KUSURSUZ DOĞRULAMA ALGORİTMASI
# ─────────────────────────────────────────────
DIGIT_OPTS = {'O': ['0'], 'Q': ['0'], 'U': ['0'], 'D': ['0'], 'I': ['1', '7'], 'L': ['1'], 'J': ['1', '7'],
              'Z': ['7', '2'],
              'E': ['3'], 'S': ['5'], 'G': ['6'], 'T': ['7'], 'F': ['7'], 'Y': ['7', '0'], 'B': ['8'], 'P': ['9'],
              'N': ['7']}
HARF_OPTS = {'0': ['O'], '1': ['I'], '2': ['Z'], '5': ['S'], '6': ['G'], '8': ['B'], '4': ['A'], '7': ['T', 'J']}
PLAKA_RE = re.compile(r'^\d{2}[A-Z]{1,3}\d{2,4}$')


def char_d(c): return DIGIT_OPTS.get(c, [c]) if not c.isdigit() else [c]


def char_h(c): return HARF_OPTS.get(c, [c]) if not c.isalpha() else [c]


def deduplicate(text):
    n = len(text)
    for length in range(2, n // 2 + 1):
        if n % length == 0 and text[:length] * (n // length) == text: return text[:length]
    for split in range(n // 2 + 1, 3, -1):
        prefix, suffix = text[:split], text[split:]
        if suffix and prefix.startswith(suffix[:min(3, len(suffix))]): return prefix
    return text


def score_aday(orijinal, aday, hbitis):
    hlen = hbitis - 2
    son_uzunluk = len(aday) - hbitis
    donusum = 0
    if len(orijinal) == len(aday):
        for o, a in zip(orijinal, aday):
            if o != a: donusum += 1
    return (hlen, son_uzunluk, -donusum)


def duzelt_ve_dogrula(text):
    text_clean = deduplicate(''.join(text.upper().split()))
    text_clean = re.sub(r'[^A-Z0-9]', '', text_clean)

    match = re.search(r'\d', text_clean)
    if match:
        ilk_rakam_index = match.start()
        if ilk_rakam_index <= 2:
            text_clean = text_clean[ilk_rakam_index:]

    if not 5 <= len(text_clean) <= 10: return None

    adaylar = []
    for hlen in [3, 2, 1]:
        hbitis = 2 + hlen
        if hbitis >= len(text_clean): continue
        opts = []
        for i, c in enumerate(text_clean):
            if i < 2:
                opts.append(char_d(c))
            elif i < hbitis:
                opts.append(char_h(c))
            else:
                opts.append(char_d(c))
        for combo in product(*opts):
            aday = ''.join(combo)
            if PLAKA_RE.match(aday):
                skor = score_aday(text_clean, aday, hbitis)
                adaylar.append((skor, aday))
    if not adaylar: return None
    adaylar.sort(key=lambda x: x[0], reverse=True)
    return adaylar[0][1]


# ─────────────────────────────────────────────
# 1. OCR İŞÇİSİ (TRAFİK POLİSİ İLE)
# ─────────────────────────────────────────────
ocr_kuyrugu = queue.Queue(maxsize=3)
plaka_hafizasi = {}


def ocr_iscisi(reader):
    print("\n[BİLGİ] OCR İşçisi Hazır! (CPU Modu & 1 Thread Lock Aktif)")
    while True:
        gorev = ocr_kuyrugu.get()
        if gorev is None: break

        track_id, plaka_crop = gorev
        try:
            h, w = plaka_crop.shape[:2]
            if w > 0:
                hedef_genislik = 400
                oran = hedef_genislik / float(w)
                hedef_yukseklik = int(h * oran)
                if oran > 1:
                    plaka_crop = cv2.resize(plaka_crop, (hedef_genislik, hedef_yukseklik),
                                            interpolation=cv2.INTER_CUBIC)
                else:
                    plaka_crop = cv2.resize(plaka_crop, (hedef_genislik, hedef_yukseklik), interpolation=cv2.INTER_AREA)

            gray = cv2.cvtColor(plaka_crop, cv2.COLOR_BGR2GRAY)

            ocr_res = reader.readtext(gray, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', width_ths=1.0)

            ham_metin = "".join([res[1] for res in ocr_res])

            if len(ham_metin) > 3:
                print(f"🔍 [OCR OKUDU] -> {ham_metin}")

            ham_dd = deduplicate(ham_metin)
            duzeltilmis = duzelt_ve_dogrula(ham_dd)

            if duzeltilmis:
                plaka_hafizasi[track_id] = duzeltilmis
                print(f"✅ [HEDEF KİLİTLENDİ] ID: {track_id} | Plaka: {duzeltilmis}")

        except Exception as e:
            print(f"⚠️ [OCR İŞÇİSİ HATASI] {e}")
        finally:
            ocr_kuyrugu.task_done()


# ─────────────────────────────────────────────
# 2. THREAD YAYIN İŞÇİSİ
# ─────────────────────────────────────────────
stream_kuyrugu = queue.Queue(maxsize=2)


def yayin_iscisi():
    client_socket = None
    if not YAYIN_YAP: return

    try:
        print(f"[AĞ] {SUNUCU_IP}:{SUNUCU_PORT} adresine bağlanılıyor...")
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((SUNUCU_IP, SUNUCU_PORT))
        print("[AĞ] Bağlantı başarılı! Asenkron Görüntü Aktarımı Aktif.")
    except Exception as e:
        print(f"[AĞ HATA] Sunucuya bağlanılamadı: {e}. Yayinsiz devam ediliyor.")
        return

    while True:
        frame = stream_kuyrugu.get()
        if frame is None: break

        try:
            stream_frame = cv2.resize(frame, (STREAM_WIDTH, STREAM_HEIGHT))
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_KALITE]
            ret_enc, buffer = cv2.imencode('.jpg', stream_frame, encode_param)

            if ret_enc:
                data = pickle.dumps(buffer)
                size_pack = struct.pack("!Q", len(data))
                client_socket.sendall(size_pack + data)
        except Exception as e:
            print(f"[AĞ HATA] Yayın koptu: {e}")
            break
        finally:
            stream_kuyrugu.task_done()

    if client_socket:
        client_socket.close()


# ─────────────────────────────────────────────
# ANA SİSTEM (MAIN)
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("[BİLGİ] YOLO ve Görüntü İşleme Sistemleri Başlatılıyor...")

    import warnings

    warnings.filterwarnings("ignore")

    # OCR Motoru Yükleniyor (CPU İle)
    reader = easyocr.Reader(['en'], gpu=False)

    # YOLO Motoru Yükleniyor (TensorRT)
    model = YOLO(YOLO_MODEL_YOLU)

    # İŞÇİLERİ BAŞLAT
    worker_ocr = threading.Thread(target=ocr_iscisi, args=(reader,), daemon=True)
    worker_ocr.start()

    worker_stream = threading.Thread(target=yayin_iscisi, daemon=True)
    worker_stream.start()

    cap = cv2.VideoCapture(USB_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)

    if not cap.isOpened():
        raise RuntimeError("Kamera açılamadı!")

    print("✅ Sistem Uçuşa Hazır. Çıkmak için terminalde Ctrl+C yap.")

    frame_sayac = 0
    fps_sayac = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame_sayac += 1

            # GPU Trafik Polisi - YOLO Çalışırken kilitler
            with gpu_lock:
                results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    if box.id is None: continue
                    track_id = int(box.id[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])

                    if conf <= 0.5: continue

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    if track_id in plaka_hafizasi:
                        okunan_plaka = plaka_hafizasi[track_id]
                        cv2.putText(frame, f"{okunan_plaka}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                                    (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, f"Okunuyor...", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255),
                                    2)

                        if not ocr_kuyrugu.full():
                            w, h = x2 - x1, y2 - y1
                            crop_x1 = max(0, x1 + int(w * 0.12))
                            crop_x2 = min(frame.shape[1], x2 - int(w * 0.02))
                            crop_y1 = max(0, y1 + int(h * 0.12))
                            crop_y2 = min(frame.shape[0], y2 - int(h * 0.12))

                            plaka_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                            if plaka_crop.size > 0:
                                ocr_kuyrugu.put((track_id, plaka_crop.copy()))

            if frame_sayac % 30 == 0:
                gecen = time.time() - fps_sayac
                fps = 30 / gecen if gecen > 0 else 0
                fps_sayac = time.time()
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

            if frame_sayac % 300 == 0:
                aktif_id_listesi = [int(box.id[0]) for r in results for box in r.boxes if box.id is not None]
                silinecekler = [tid for tid in plaka_hafizasi if tid not in aktif_id_listesi]
                for tid in silinecekler: del plaka_hafizasi[tid]

            if not stream_kuyrugu.full():
                stream_kuyrugu.put(frame.copy())

    except KeyboardInterrupt:
        print("\n[BİLGİ] Kullanıcı tarafından durduruldu.")
    finally:
        print("[BİLGİ] Sistem güvenli kapatılıyor...")
        ocr_kuyrugu.put(None)
        stream_kuyrugu.put(None)
        cap.release()