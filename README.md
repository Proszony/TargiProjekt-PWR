Projekt zespołowy
=================

Repozytorium zawiera aplikację do **booth analytics** na wspólnej mapie 2D hali. System synchronizuje wiele kamer, śledzi osoby lokalnie w każdej kamerze, deduplikuje je wyłącznie w obszarach overlapu i liczy proste, użyteczne metryki stoisk.

Najważniejsze możliwości
------------------------
- odbiór kamer UDP lub lokalnych plików MP4,
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

Konfiguracja
------------
- `config/project.json`: projekt, playback sync, overlap dedup, analytics.
- `config/cameras/*.json`: źródła kamer, tracking, kalibracja, coverage.
- `config/venue.json`: mapa hali i strefy / stoiska.

Jeżeli nie wczytasz pliku mapy, aplikacja użyje pustego tła z siatką i nadal pozwoli kalibrować oraz rysować strefy w przestrzeni 2D.

Testy
-----
```bash
python -m unittest discover -s tests
```
