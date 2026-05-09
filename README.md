Projekt zespołowy
=================

Repozytorium zawiera prototyp systemu monitoringu targów oparty o wspólną mapę 2D hali. Raspberry Pi wysyła obraz z kamery przez UDP, a aplikacja na komputerze wykrywa osoby, śledzi ich pozycje, rzutuje je do przestrzeni hali i liczy metryki dla stref stoisk.

Najważniejsze elementy
----------------------
- odbiór strumienia UDP z kamery,
- detekcja osób YOLO,
- tracking wieloobiektowy przez Ultralytics Track Mode z BoT-SORT albo ByteTrack,
- kalibracja kamera -> mapa hali przez homografię,
- definiowanie stref na mapie 2D,
- liczenie obłożenia stref, czasu pobytu i powrotów.

Detektor a tracker
------------------
- Detektor odpowiada za znalezienie osoby na pojedynczej klatce.
- Tracker odpowiada za utrzymanie tego samego ID między klatkami.
- Stabilne ID nie wynika z samego mocniejszego modelu detekcji. W praktyce trzeba połączyć detekcję z MOT i pamięcią trackera.

`Robust detection`
------------------
Opcja `Robust detection` włącza augmented inference po stronie detektora. To nadal jest ten sam model, ale uruchamiany w droższym trybie testowym z dodatkowymi transformacjami obrazu.

- `Off`: szybsze działanie, mniejsze zużycie GPU.
- `On`: wolniejsze działanie, ale zwykle lepsza detekcja małych osób, profilu i trudniejszych ujęć monitoringu.

Ta opcja poprawia jakość detekcji, ale nie zastępuje pamięci trackera.

Uruchomienie
------------
1. Zainstaluj zależności:
   ```bash
   pip install -r requirements.txt
   ```
2. Uruchom aplikację:
   ```bash
   python raspberry-feed-yolo.py
   ```

Streaming z Raspberry Pi
------------------------
Do przesyłu wykorzystano `ffmpeg`, które należy wcześniej zainstalować na Raspberry Pi.

```bash
#!/bin/bash
ip=$1
port=$2

ffmpeg -f v4l2 -framerate 15 -video_size 640x480 -i /dev/video0 \
  -c:v h264_v4l2m2m -b:v 1M -f mpegts udp://$ip:$port
```

Konfiguracja
------------
- `config/venue.json`: mapa hali i strefy.
- `config/cameras/camera-1.json`: konfiguracja kamery, progi trackingu i kalibracja.
- `config/trackers/botsort.yaml`: bazowa konfiguracja BoT-SORT.
- `config/trackers/bytetrack.yaml`: bazowa konfiguracja ByteTrack.

Jeżeli nie wczytasz pliku mapy, aplikacja użyje pustego tła z siatką i nadal pozwoli kalibrować oraz rysować strefy w przestrzeni 2D.

Rekomendowane presety
---------------------
RTX 3060 Ti 8 GB:

- tryb zbalansowany: `yolo26m.pt` + `BoT-SORT` + `Inference size 640`
- trudny pojedynczy clip MP4: `yolo26l.pt` + `BoT-SORT` + `Inference size 640` albo `768`
- lżejszy live stream: `yolo26s.pt` + `ByteTrack` albo `BoT-SORT`, zależnie od sceny

Domyślna rekomendacja dla tego projektu to `yolo26m.pt + BoT-SORT`, z wyłączonym `Robust detection` na żywo i opcjonalnym włączeniem go dla trudnych nagrań offline.

Testy
-----
```bash
python -m unittest discover -s tests
```
