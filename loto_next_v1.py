#!/usr/bin/env python3
"""
Loto 7/39 - Nicolas Bourbaki


Postupak:
1. Uči distribuciju zbira iz prethodnih kola.
2. Uči distribuciju raspona uslovljenu zbirom i prethodnim kolima.
3. Izračunava zajedničku distribuciju zbira i raspona.
4. Tačno prebrojava sedmorke svakog ostvarivog profila.
5. Rangira profile prema masi po pojedinačnoj sedmorki.
6. Ispisuje jednu deterministički izabranu NEXT sedmorku.
"""

import argparse
import csv
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from distribution_regressor import DistributionRegressor


DEFAULT_CSV = Path(
    "/data/loto7_4692_k77.csv"
    # "/data/loto7_4692_k77_loto_2968.csv"
    # "/data/loto7_4692_k77_loto_plus_1724.csv"
)
SUM_VALUES = np.arange(28, 253)
RANGE_VALUES = np.arange(6, 39)
TOTAL_COMBINATIONS = math.comb(39, 7)


def load_csv(path):
    draws = []

    with path.open(encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), 1):
            if not row or not any(value.strip() for value in row):
                continue

            try:
                numbers = [int(value.strip()) for value in row]
            except ValueError as error:
                raise ValueError(
                    f"Red {line_number}: očekujem cijele brojeve."
                ) from error

            valid = (
                len(numbers) == 7
                and len(set(numbers)) == 7
                and all(1 <= number <= 39 for number in numbers)
            )
            if not valid:
                raise ValueError(
                    f"Red {line_number}: potrebno je tačno "
                    "7 različitih brojeva između 1 i 39."
                )

            draws.append(sorted(numbers))

    if not draws:
        raise ValueError("CSV ne sadrži izvlačenja.")

    return np.asarray(draws, dtype=np.int64)


def build_dataset(draws, lags):
    descriptors = []

    for numbers in draws:
        digit_set = {
            int(character)
            for number in numbers
            for character in f"{number:02d}"
        }

        descriptors.append([
            int(numbers.sum()),
            int(numbers[-1] - numbers[0]),
            *np.diff(numbers),
            int(np.sum(numbers % 2)),
            len(digit_set),
        ])

    descriptors = np.asarray(descriptors, dtype=np.float64)

    # Za cilj t koristimo isključivo redove pre t.
    inputs = np.asarray([
        descriptors[t - lags:t].ravel()
        for t in range(lags, len(draws) + 1)
    ])

    targets = descriptors[lags:, :2].astype(np.int64)

    return inputs[:-1], targets, inputs[-1:]


def count_profiles():
    """Tačan broj sedmorki za svaki par (zbir, raspon)."""
    counts = np.zeros((253, 39), dtype=np.int64)

    for minimum in range(1, 34):
        # DP broji izbor pet unutrašnjih brojeva.
        dp = np.zeros((6, 253), dtype=np.int64)
        dp[0, 0] = 1

        for maximum in range(minimum + 1, 40):
            offset = minimum + maximum
            remaining_size = 253 - offset

            if remaining_size > 0:
                counts[offset:, maximum - minimum] += (
                    dp[5, :remaining_size]
                )

            # Ovaj maksimum postaje unutrašnji broj tek
            # za narednu vrednost maksimuma.
            for selected in range(5, 0, -1):
                dp[selected, maximum:] += (
                    dp[selected - 1, :-maximum]
                )

    if int(counts.sum()) != TOTAL_COMBINATIONS:
        raise RuntimeError("Neispravno ukupno prebrojavanje.")

    for span in RANGE_VALUES:
        expected = (
            (39 - int(span))
            * math.comb(int(span) - 1, 5)
        )
        if int(counts[:, span].sum()) != expected:
            raise RuntimeError("Neispravno prebrojavanje raspona.")

    return counts[28:253, 6:39]


def fit_distribution(x, y, trees):
    if np.min(y) == np.max(y):
        raise ValueError(
            "Ciljna veličina nema dovoljno različitih vrednosti."
        )

    model = DistributionRegressor(
        n_bins=int(np.max(y) - np.min(y)) + 1,
        use_base_model=False,
        monte_carlo_training=False,
        output_smoothing=0,
        n_estimators=trees,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        n_jobs=2,
        verbosity=-1,
        random_state=739,
    )
    model.fit(x, y)
    return model


def predict_pmf(model, x, support):
    grids, masses, _ = model.predict_distribution(x)

    grids = np.asarray(grids, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)

    if (
        grids.shape != masses.shape
        or not np.isfinite(grids).all()
        or not np.isfinite(masses).all()
        or np.any(masses < -1e-10)
    ):
        raise RuntimeError("Model je vratio neispravnu distribuciju.")

    # Mreža ima korak 1; uklanjamo numerička odstupanja.
    indices = np.rint(grids).astype(np.int64) - int(support[0])

    if np.any(indices < 0) or np.any(indices >= len(support)):
        raise RuntimeError("Vrednost izvan dozvoljenog opsega.")

    pmf = np.zeros((len(x), len(support)), dtype=np.float64)

    for row in range(len(x)):
        np.add.at(
            pmf[row],
            indices[row],
            np.maximum(masses[row], 0),
        )

    totals = pmf.sum(axis=1, keepdims=True)

    if np.any(totals <= 0):
        raise RuntimeError("Model je vratio praznu distribuciju.")

    return pmf / totals


def choose_combination(target_sum, target_span, expected_count):
    """
    Bira srednju sedmorku u leksikografskom poretku profila.
    Sve sedmorke tog profila imaju istu ocenu ovog modela.
    """

    @lru_cache(maxsize=None)
    def ways(start, stop, left, remaining_sum):
        if left == 0:
            return int(remaining_sum == 0)

        if stop - start < left:
            return 0

        minimum_sum = left * (2 * start + left - 1) // 2
        maximum_sum = left * (2 * stop - left - 1) // 2

        if not minimum_sum <= remaining_sum <= maximum_sum:
            return 0

        return sum(
            ways(
                number + 1,
                stop,
                left - 1,
                remaining_sum - number,
            )
            for number in range(start, stop - left + 1)
        )

    groups = []

    for minimum in range(1, 40 - target_span):
        maximum = minimum + target_span
        remaining_sum = target_sum - minimum - maximum

        count = ways(
            minimum + 1,
            maximum,
            5,
            remaining_sum,
        )

        if count:
            groups.append((
                minimum,
                maximum,
                remaining_sum,
                count,
            ))

    total = sum(group[3] for group in groups)

    if total == 0 or total != expected_count:
        raise RuntimeError(
            "Ne slažu se nezavisna prebrojavanja profila."
        )

    rank = (total - 1) // 2

    for minimum, maximum, remaining_sum, count in groups:
        if rank >= count:
            rank -= count
            continue

        result = [minimum]
        start = minimum + 1

        for left in range(5, 0, -1):
            for number in range(start, maximum - left + 1):
                block_size = ways(
                    number + 1,
                    maximum,
                    left - 1,
                    remaining_sum - number,
                )

                if rank < block_size:
                    result.append(number)
                    remaining_sum -= number
                    start = number + 1
                    break

                rank -= block_size
            else:
                raise RuntimeError("Neuspešan izbor NEXT kombinacije.")

        result.append(maximum)

        if not (
            len(result) == 7
            and len(set(result)) == 7
            and all(1 <= number <= 39 for number in result)
            and sum(result) == target_sum
            and result[-1] - result[0] == target_span
        ):
            raise RuntimeError("NEXT nije prošao završnu proveru.")

        return result

    raise RuntimeError("Nije pronađena NEXT kombinacija.")


def main():
    parser = argparse.ArgumentParser(
        description="Loto 7/39: distribucije → jedna NEXT sedmorka."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
    )
    parser.add_argument("--lags", type=int, default=3)
    parser.add_argument("--trees", type=int, default=80)
    parser.add_argument(
        "--newest-first",
        action="store_true",
        help="Koristi samo ako je najnovije izvlačenje prvi red.",
    )

    args = parser.parse_args()

    if args.lags < 1 or args.trees < 1:
        parser.error("lags i trees moraju biti pozitivni.")

    try:
        draws = load_csv(args.csv)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    if args.newest_first:
        draws = draws[::-1].copy()

    if len(draws) - args.lags < 200:
        parser.error("Potrebno je najmanje 200 trening primera.")

    x_train, targets, next_x = build_dataset(draws, args.lags)
    sum_targets = targets[:, 0]
    span_targets = targets[:, 1]

    print(f"CSV: {args.csv}")
    print(f"Izvlačenja: {len(draws)}")
    print(
        "Poslednje:",
        " ".join(f"{number:02d}" for number in draws[-1]),
    )

    print("\n[1/4] Tačno prebrojavanje sedmorki...", flush=True)
    profile_counts = count_profiles()

    print("[2/4] Učenje distribucije zbira...", flush=True)
    sum_model = fit_distribution(
        x_train,
        sum_targets,
        args.trees,
    )
    sum_pmf = predict_pmf(
        sum_model,
        next_x,
        SUM_VALUES,
    )[0]

    print("[3/4] Učenje raspona uslovljenog zbirom...", flush=True)

    # Stvarni zbir je uslov tokom treninga.
    conditional_train = np.column_stack((
        x_train,
        sum_targets,
    ))

    span_model = fit_distribution(
        conditional_train,
        span_targets,
        args.trees,
    )

    # Za NEXT razmatramo sve moguće zbirove.
    conditional_next = np.column_stack((
        np.repeat(next_x, len(SUM_VALUES), axis=0),
        SUM_VALUES,
    ))

    span_given_sum = predict_pmf(
        span_model,
        conditional_next,
        RANGE_VALUES,
    )

    # P(zbir, raspon | istorija)
    # = P(zbir | istorija) * P(raspon | zbir, istorija)
    joint = sum_pmf[:, None] * span_given_sum

    possible = profile_counts > 0
    removed_mass = float(joint[~possible].sum())
    joint[~possible] = 0

    mass = float(joint.sum())
    if not np.isfinite(mass) or mass <= 0:
        raise RuntimeError("Nema mase na ostvarivim profilima.")

    joint /= mass

    # Pretpostavka: jednaka masa sedmorkama istog profila.
    score = np.divide(
        joint,
        profile_counts,
        out=np.zeros_like(joint),
        where=possible,
    )

    best_sum_index, best_span_index = np.unravel_index(
        int(np.argmax(score)),
        score.shape,
    )

    selected_sum = int(SUM_VALUES[best_sum_index])
    selected_span = int(RANGE_VALUES[best_span_index])
    selected_count = int(
        profile_counts[best_sum_index, best_span_index]
    )

    print("[4/4] Izbor i provera NEXT sedmorke...", flush=True)

    next_numbers = choose_combination(
        selected_sum,
        selected_span,
        selected_count,
    )

    print("\n" + "=" * 58)
    print(
        "NEXT:",
        " ".join(f"{number:02d}" for number in next_numbers),
    )
    print("=" * 58)
    print(f"Zbir: {selected_sum}")
    print(f"Raspon: {selected_span}")
    print(f"Sedmorki tog profila: {selected_count:,}")
    print(
        "Masa profila po modelu:",
        f"{joint[best_sum_index, best_span_index]:.6%}",
    )
    print(
        "Ocena po sedmorki:",
        f"{score[best_sum_index, best_span_index]:.10g}",
    )
    print(f"Odbačena masa nemogućih profila: {removed_mass:.3%}")
    print(
        "Unutar profila izabrana je srednja sedmorka;"
        " model ih međusobno ne razlikuje."
    )


if __name__ == "__main__":
    main()



"""
CSV: /data/loto7_4692_k77.csv
Izvlačenja: 4692
Poslednje: 03 05 12 14 15 23 31

[1/4] Tačno prebrojavanje sedmorki... Tacno brojanje profila...
[2/4] Učenje distribucije zbira... Ucenje P(zbir | prethodna kola)...
[3/4] Učenje raspona uslovljenog zbirom... Ucenje P(raspon | zbir, prethodna kola)...
[4/4] Izbor i provera NEXT sedmorke...


DISTRIBUCIJA ZBIRA
  Ocekivana vrijednost: 139.592
  Medijana: 140
  Centralni interval 50%: 119-160
  Centralni interval 80%: 102-176
  Centralni interval 95%: 84-198

DISTRIBUCIJA RASPONA
  Ocekivana vrijednost: 29.685
  Medijana: 31
  Centralni interval 50%: 26-34
  Centralni interval 80%: 22-36
  Centralni interval 95%: 18-38


==========================================================
NEXT: 01 02 03 x y z 26
==========================================================
Zbir: 48
Raspon: 25
Sedmorki tog profila: 1
Masa profila po modelu: 0.042627%
Ocena po sedmorki: 0.0004262714329
Odbačena masa nemogućih profila: 0.129%
Unutar profila izabrana je srednja sedmorka; model ih međusobno ne razlikuje.
Modeli imaju naucenu masu samo u opsegu ciljeva vidjenih u treningu.
"""



"""
CSV: /data/loto7_4692_k77_loto_2968.csv
Izvlačenja: 2968
Poslednje: 01 05 09 17 19 27 34

[1/4] Tačno prebrojavanje sedmorki...
[2/4] Učenje distribucije zbira...
[3/4] Učenje raspona uslovljenog zbirom...
[4/4] Izbor i provera NEXT sedmorke...

==========================================================
NEXT: 01 02 03 x y z 30
==========================================================
Zbir: 52
Raspon: 29
Sedmorki tog profila: 1
Masa profila po modelu: 0.023215%
Ocena po sedmorki: 0.0002321490182
Odbačena masa nemogućih profila: 0.083%
Unutar profila izabrana je srednja sedmorka; model ih međusobno ne razlikuje.
"""



"""
CSV: /data/loto7_4692_k77_loto_plus_1724.csv
Izvlačenja: 1724
Poslednje: 03 05 12 14 15 23 31

[1/4] Tačno prebrojavanje sedmorki...
[2/4] Učenje distribucije zbira...
[3/4] Učenje raspona uslovljenog zbirom...
[4/4] Izbor i provera NEXT sedmorke...

==========================================================
NEXT: 01 02 03 x y z 26
==========================================================
Zbir: 48
Raspon: 25
Sedmorki tog profila: 1
Masa profila po modelu: 0.074367%
Ocena po sedmorki: 0.0007436683743
Odbačena masa nemogućih profila: 0.691%
Unutar profila izabrana je srednja sedmorka; model ih međusobno ne razlikuje.
"""



"""
Nicolas Bourbaki - 1935, Henri Cartan, Jean Dieudonné, Claude Chevalley, André Weil.
"""



"""
Da bolje povežemo distribucije s izborom NEXT sedmorke. 
Primena bi bila u preciznom opisivanju i prebrojavanju kombinacija.
Nicolas Bourbaki je zajedničko ime grupe matematičara, a ne pojedinac. 
Njihova dela obuhvataju teoriju skupova, algebru, integraciju i druge oblasti. 

Za Loto 7/39 tri primene. 
1. Grupisanje sedmorki prema strukturi
Svaku sedmorku predstavimo profilom:
(zbir, raspon, broj neparnih, raspored po intervalima, pokrivene cifre)
Sedmorke s istim profilom pripadaju istoj klasi. 
Tako model prvo procenjuje distribuciju profila, a zatim biramo konkretnu sedmorku unutar odabrane klase.
2. Tačno prebrojavanje klasa — najkorisniji dio
Klase nisu jednako velike. Profil može biti čest zato što mu pripada mnogo kombinacija.
To ispravlja važnu slabost: najverovatniji zbir nije dovoljan da odredi najbolje rangiranu pojedinačnu kombinaciju.
3. Zajednička distribucija osobina
Zbir i raspon su povezani. 

Zatim tačnim prebrojavanjem isključujemo nemoguće profile i raspodjeljujemo masu među ostvarivim sedmorkama.
- primieniti klase kombinacija + tačno prebrojavanje + zajedničku distribuciju zbira i raspona. 
To daje konkretan put od distribucije do NEXT. 

Ako više sedmorki ima isti profil, ovaj model ih još ne razlikuje 
— za izbor jedne treba dodatna osobina ili jasno pravilo razrješenja jednakih ocena.
Za taj posao dovoljni su konačna kombinatorika i uslovna verovatnoća; 
ne treba uvoditi težu Bourbakijevu teoriju samo zbog imena.



Kod uči P(zbir) x P(raspon | zbir), 
deli verovatnoću profila tačnim brojem njegovih sedmorki i ispisuje jednu NEXT kombinaciju. 
Među jednako rangiranim sedmorkama bira srednju u sortiranom poretku.



Tačno prebrojavanje je provereno: ukupno 15.380.937 sedmorki. 

distribucija zbira → raspon uslovljen zbirom → prebrojavanje profila → jedna NEXT sedmorka.
s dodatnim proverama i preglednijim ispisom



v1 rezultat otkriva grešku u načinu rangiranja. 
Deljenje mase profila brojem njegovih sedmorki može podići retke, ekstremne profile. 
Ova kombinacija ima zbir 48 — upravo donju granicu zbirova. 
Model je na rubu raspodele dodelio masu profilu s vrlo malo kombinacija, 
pa ga je račun masa / broj kombinacija izbacio na vrh.
Zatim pravilo „uzmi srednju sedmorku” samo bira predstavnika tog profila; 
ne procjenjuje pojedinačne brojeve.

Ispravka u v2 je da prvo biramo ostvariv profil prema njegovoj zajedničkoj masi, bez tog dijeljenja. 
Time popravljamo izbor profila. Za smislen izbor jedne sedmorke unutar njega trebamo dodatno modelirati strukturu, recimo razmake između brojeva. 
Sam zbir i raspon nisu dovoljni.
Ne popravljati ručnom zabranom brojeva 1-7 — problem je u kriteriju.
"""
