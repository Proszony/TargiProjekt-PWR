Projekt zespołowy
=================

Repozytorium zawiera aplikację do **booth analytics** na wspólnej mapie 2D hali. System synchronizuje wiele kamer, śledzi osoby lokalnie w każdej kamerze, deduplikuje je wyłącznie w obszarach overlapu i liczy proste, użyteczne metryki stoisk.

Najważniejsze możliwości
------------------------
- odbiór kamer UDP lub lokalnych plików MP4,
- tryb rozproszony server/worker dla kamer uruchamianych na innych hostach,
- detekcja osób YOLO,
- lokalny tracking przez Ultralytics Track Mode z BoT-SORT,
- kalibracja kamera -> mapa hali przez homografię,
- definiowanie stoisk i stref na mapie 2D,
- overlap-only deduplication, żeby ograniczyć podwójne liczenie w nakładających się kamerach,
- liczenie:
  - aktualnego occupancy,
  - liczby wizyt,
  - średniego czasu przy stoisku,
  - mediany czasu przy stoisku,
  - peak occupancy,
  - timeline occupancy.

Semantyka produktu
------------------
- To nie jest już produkt do pełnej identyfikacji osoby między wszystkimi kamerami.
- Cross-camera matching służy tylko do **deduplikacji countingu w overlapie**.
- Poza overlapem aplikacja nie obiecuje event-wide continuity tej samej osoby.

Uruchomienie
------------
1. Zainstaluj zależności:
   ```bash
   pip install -r requirements.txt
   ```
2. Uruchom aplikację:
   ```bash
   python main.py
   ```

Tryb rozproszony
----------------
- Serwer z UI i analytics uruchamiasz standardowo:
  ```bash
  python main.py
  ```
- Worker dla jednej kamery uruchamiasz osobno:
  ```bash
  python -m app.main --mode worker --camera-id camera-1 --server-host 192.168.1.10 --server-port 6100
  ```
- Kamera przypisana do workera musi mieć w konfiguracji `runtime_mode = remote`.
- Konfiguracja kamery, mapy, synchronizacji i runtime jest wysyłana z serwera do workera po połączeniu.
- Obraz mapy z `venue.json` jest pobierany przez workera, jeżeli plik istnieje po stronie serwera. Nowo wybrane mapy są zapisywane w projekcie jako ścieżki względne.

Konfiguracja
------------
- `config/project.json`: projekt, playback sync, overlap dedup, analytics.
- `config/cameras/*.json`: źródła kamer, tracking, kalibracja, coverage.
- `config/venue.json`: mapa hali i strefy / stoiska.
- `project.distributed_runtime`: ustawienia serwera TCP i heartbeatów workerów.
- `camera.runtime_mode` / `camera.remote_worker_id`: przypisanie kamery do trybu lokalnego lub zdalnego workera.
- Ścieżki plików w konfiguracji powinny być względne do katalogu projektu i zapisywane z ukośnikami `/`, co działa na Windows i Linux.

Jeżeli nie wczytasz pliku mapy, aplikacja użyje pustego tła z siatką i nadal pozwoli kalibrować oraz rysować strefy w przestrzeni 2D.

Testy
-----
```bash
python -m unittest discover -s tests
```
