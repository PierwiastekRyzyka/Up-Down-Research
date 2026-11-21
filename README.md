# Poszukiwanie nieefektywności na 5-minutowych rynkach BTC na Polymarkecie

Badanie, czy na rynkach „Bitcoin Up or Down” (okno 5 minut, rozliczenie TWAP-60s) występują
nieefektywności cenowe, na których da się systematycznie zarabiać po uwzględnieniu opłat,
opóźnień i sposobu składania zleceń.

Opis wyników: [artykuł](https://krzychupll.github.io/new/projekty/polymarket-5m/) ·
pełny raport: [`docs/raport.pdf`](docs/raport.pdf)

## Wyniki

Próba: 2 680 rynków (co trzeci rynek, 23.10–19.11.2026), 4,0 mln transakcji w oknach rynków,
ok. 35 tys. portfeli, 61 mln USD obrotu.

| | |
|---|---|
| Zgodność zrekonstruowanej reguły rozliczenia z oficjalnymi wynikami | 98,4% (dosłowne odczytanie opisu: 87,0%) |
| Brier score: cena rynkowa / model | 0,1543 / 0,1546 |
| Część luki na otwarciu uwzględniona w cenie | 63% |
| Czas, po którym ruch BTC przestaje dawać przewagę nad ceną | ok. 3 s |
| Wynik takerów przed opłatami | −0,015¢ na akcję |
| Portfele z istotnym zyskiem w obu połowach próby | 13 (wszystkie kupują faworytów po ok. 0,99 USD) |
| Reguła faworyta, zlecenie limit (bez opłaty) | +2,63¢ na akcję, t = 8,4 |
| Reguła faworyta, FOK (opłata 0,07) | +2,02¢ na akcję, t = 4,9 |

Wniosek: ceny są efektywne z dokładnością do ok. 3 sekund. Jedyna nieefektywność, która
przetrwała test poza próbą, dotyczy prawie pewnych faworytów i wynika z kształtu opłaty
`0,07 · p · (1 − p)`, która przy cenie 0,99 jest bliska zera. Pojemność tej strategii jest mała.

## Metoda

**Reguła rozliczenia.** Up wygrywa, gdy średnia cena z ostatnich 60 s okna jest nie niższa
niż średnia z 60 s przed jego startem (B). Wariant ten wybrano empirycznie spośród kilku
odczytań opisu rynku (`analysis/01_settlement_rule.py`).

**Model.** Cena BTC w skali sekund jako ruch Browna bez dryfu, σ ze zrealizowanej
zmienności z ostatniej godziny. Przed ostatnią minutą:

```
P(Up) = Φ( ln(S_t / B) / (σ · √(260 − t)) )
```

W ostatniej minucie wzór uwzględnia zrealizowaną część średniej (`src/pm5m/model.py`).
Model nie ma parametrów dopasowanych do wyników, a wzory są sprawdzone symulacją
Monte Carlo (`tests/test_model.py`).

**Test poza próbą.** Rynki podzielono na dwie połowy po medianie czasu startu. Wynik uznaje
się za wiarygodny tylko wtedy, gdy utrzymuje się w obu.

**Opóźnienie.** Każdą przewagę sprawdzono przy sygnale opóźnionym o 0–10 s, a backtest
liczono po cenach transakcji, które faktycznie odbyły się po sygnale.

## Dane

| Źródło | Zakres |
|---|---|
| Polymarket Gamma API | metadane i wyniki rynków |
| Polymarket Data API v2 | transakcje (strona takera, z portfelem) |
| data.binance.vision | BTCUSDT, świece 1 s |
| Deribit | DVOL (zmienność implikowana, wariant odporności) |

Rynki rozlicza strumień Chainlink, do którego badanie nie ma bezpośredniego dostępu. Cenę BTC
przybliżono danymi z Binance, więc wartości modelu i rekonstrukcja reguły są przybliżone.
Wszystkie 44 rozbieżności z oficjalnymi wynikami mają margines poniżej 2 punktów bazowych.

## Struktura

```
src/pm5m/
  config.py        ścieżki, okres próby, stawki opłat
  data/            pobieranie danych (Polymarket, Binance, Deribit)
  model.py         P(Up) dla reguły TWAP-60s
  panel.py         tabele rynków i panel rynek × moment w oknie
  sample.py        ładowanie próby, wartość modelu dla każdej transakcji
  strategies.py    reguła faworyta
  stats.py         Brier, regresja logistyczna, błędy klastrowane po rynku
analysis/          skrypty 01–07, po jednym na rozdział badania
results/           wyniki liczbowe (JSON)
figures/           wykresy
tests/             test modelu (Monte Carlo)
docs/raport.pdf    pełny raport
```

## Uruchomienie

```bash
make install     # pip install -e ".[dev]"
make data        # pobranie danych, ok. 20 min
make panel       # budowa tabel analitycznych
make analysis    # analysis/01–07 -> results/, figures/
make test
```

Wymagany Python 3.10+.

## Ograniczenia

- Cena rynkowa to mediana transakcji, a nie kwotowania bid/ask. Realny koszt wejścia może
  być wyższy niż przyjęty poślizg 1¢.
- Wyniki zleceń limit są rekonstruowane z wypełnień innych uczestników. Bez historii księgi
  zleceń nie da się ustalić miejsca w kolejce ani częstości wypełnień własnych zleceń.
- Próba obejmuje co trzeci rynek i tylko BTC w oknie 5-minutowym. Polymarket prowadzi też
  rynki 15-minutowe, godzinowe, 4-godzinne i dzienne oraz rynki na inne kryptowaluty.
- Stawka opłaty 0,07 obowiązywała w okresie próby. Od 20.11.2026 23:59 UTC dla BTC 5m i 15m
  wynosi 0,05, a zasady rabatów dla makerów są inne.

Materiał ma charakter badawczy i nie jest rekomendacją inwestycyjną.
