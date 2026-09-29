#!/usr/bin/env python3
"""Loto 7/39: distribucije, struktura i jedna NEXT kombinacija."""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from distribution_regressor import DistributionRegressor

DEFAULT_CSV = "/data/loto7_4692_k77.csv"
# DEFAULT_CSV = "/data/loto7_4692_k77_loto_2968.csv"
# DEFAULT_CSV = "/data/loto7_4692_k77_loto_plus_1724.csv"

ZBIROVI = np.arange(28, 253)
RASPONI = np.arange(6, 39)
BROJEVI = np.arange(1, 40)


def ucitaj(putanja):
    izvlacenja = []
    with Path(putanja).open(encoding="utf-8-sig", newline="") as fajl:
        for red, polja in enumerate(csv.reader(fajl), 1):
            if not polja or not any(p.strip() for p in polja):
                continue
            try:
                brojevi = sorted(int(p.strip()) for p in polja)
            except ValueError as greska:
                raise ValueError(f"Red {red}: očekuju se celi brojevi.") from greska
            if (
                len(brojevi) != 7
                or len(set(brojevi)) != 7
                or not all(1 <= b <= 39 for b in brojevi)
            ):
                raise ValueError(f"Red {red}: potrebno je 7 različitih brojeva 1–39.")
            izvlacenja.append(brojevi)
    return np.asarray(izvlacenja, dtype=np.int64)


def pripremi(izvlacenja, koraci):
    osobine = []
    for brojevi in izvlacenja:
        cifre = {c for b in brojevi for c in f"{b:02d}"}
        osobine.append([
            brojevi.sum(),
            brojevi[-1] - brojevi[0],
            *np.diff(brojevi),
            np.sum(brojevi % 2),
            len(cifre),
        ])
    osobine = np.asarray(osobine, dtype=float)
    # Za cilj t koristimo samo izvlačenja pre t.
    x = np.asarray([
        osobine[t - koraci:t].ravel()
        for t in range(koraci, len(izvlacenja) + 1)
    ])
    return x[:-1], izvlacenja[koraci:], x[-1:]


def ostvarivi_profili():
    """Tačno prebrojavanje služi samo proveri ostvarivosti profila."""
    broj = np.zeros((253, 39), dtype=np.int64)
    for najmanji in range(1, 34):
        dp = np.zeros((6, 253), dtype=np.int64)
        dp[0, 0] = 1
        for najveci in range(najmanji + 1, 40):
            pomeraj = najmanji + najveci
            broj[pomeraj:, najveci - najmanji] += dp[5, :253 - pomeraj]
            for k in range(5, 0, -1):
                dp[k, najveci:] += dp[k - 1, :-najveci]
    if int(broj.sum()) != math.comb(39, 7):
        raise RuntimeError("Neispravno prebrojavanje profila.")
    return broj[28:253, 6:39] > 0


def obuci(x, y, stabla):
    if np.min(y) == np.max(y):
        raise ValueError("Nema dovoljno različitih ciljnih vrednosti.")
    model = DistributionRegressor(
        n_bins=int(np.max(y) - np.min(y)) + 1,
        use_base_model=False,
        monte_carlo_training=False,
        output_smoothing=0,
        n_estimators=stabla,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        n_jobs=2,
        verbosity=-1,
        random_state=739,
    )
    model.fit(x, y)
    return model


def raspodela(model, x, opseg):
    mreza, mase, _ = model.predict_distribution(x)
    mreza = np.asarray(mreza, dtype=float)
    mase = np.asarray(mase, dtype=float)
    if (
        mreza.shape != mase.shape
        or not np.isfinite(mreza).all()
        or not np.isfinite(mase).all()
        or np.any(mase < -1e-10)
    ):
        raise RuntimeError("Model je vratio neispravnu raspodelu.")
    indeksi = np.rint(mreza).astype(int) - int(opseg[0])
    if np.any(indeksi < 0) or np.any(indeksi >= len(opseg)):
        raise RuntimeError("Predikcija je izvan dozvoljenog opsega.")
    rezultat = np.zeros((len(x), len(opseg)))
    for red in range(len(x)):
        np.add.at(rezultat[red], indeksi[red], np.maximum(mase[red], 0))
    ukupno = rezultat.sum(axis=1, keepdims=True)
    if np.any(ukupno <= 0):
        raise RuntimeError("Model je vratio praznu raspodelu.")
    return rezultat / ukupno


def obuci_modele(x, ciljevi, stabla):
    zbirovi = ciljevi.sum(axis=1)
    rasponi = ciljevi[:, -1] - ciljevi[:, 0]

    print("Učenje distribucije zbira...", flush=True)
    modeli = [obuci(x, zbirovi, stabla)]

    print("Učenje uslovne distribucije raspona...", flush=True)
    modeli.append(obuci(np.column_stack((x, zbirovi)), rasponi, stabla))

    osnova = np.column_stack((x, zbirovi, rasponi))
    for k in range(7):
        print(f"Učenje strukture: broj {k + 1}/7...", flush=True)
        # Svaki broj zavisi i od prethodnih brojeva iste sedmorke.
        ulaz = np.column_stack((osnova, ciljevi[:, :k]))
        modeli.append(obuci(ulaz, ciljevi[:, k], stabla))
    return modeli


def izaberi(modeli, x, moguce, sirina):
    p_zbir = raspodela(modeli[0], x, ZBIROVI)[0]
    uslovi = np.column_stack((
        np.repeat(x, len(ZBIROVI), axis=0), ZBIROVI
    ))
    p_raspon = raspodela(modeli[1], uslovi, RASPONI)

    zajednicka = p_zbir[:, None] * p_raspon
    zajednicka[~moguce] = 0
    if zajednicka.sum() <= 0:
        raise RuntimeError("Nema mase na ostvarivim profilima.")
    zajednicka /= zajednicka.sum()

    # Rangiranje prema masi profila, bez deljenja brojem sedmorki.
    i, j = np.unravel_index(np.argmax(zajednicka), zajednicka.shape)
    zbir, raspon = int(ZBIROVI[i]), int(RASPONI[j])
    stanja = [(0.0, ())]

    # Ograničena pretraga zadržava najbolje delimične kombinacije.
    for k in range(7):
        ulazi = np.asarray([
            np.r_[x[0], zbir, raspon, prefiks]
            for _, prefiks in stanja
        ])
        verovatnoce = raspodela(modeli[k + 2], ulazi, BROJEVI)
        kandidati = []

        for red, (ocena, prefiks) in enumerate(stanja):
            if not prefiks:
                opcije = range(1, 40 - raspon)
            elif k == 6:
                opcije = [prefiks[0] + raspon]
            else:
                opcije = range(prefiks[-1] + 1, prefiks[0] + raspon)

            for broj in opcije:
                if prefiks and broj <= prefiks[-1]:
                    continue
                masa = float(verovatnoce[red, broj - 1])
                if masa <= 0:
                    continue

                novi = prefiks + (broj,)
                najveci = novi[0] + raspon
                if k < 6:
                    preostalo = 5 - k
                    ostatak = zbir - sum(novi) - najveci
                    if najveci - broj - 1 < preostalo:
                        continue
                    donja = preostalo * (2 * (broj + 1) + preostalo - 1) // 2
                    gornja = preostalo * (2 * najveci - preostalo - 1) // 2
                    if not donja <= ostatak <= gornja:
                        continue
                elif sum(novi) != zbir:
                    continue

                kandidati.append((ocena + math.log(masa), novi))

        if not kandidati:
            raise RuntimeError(
                "Pretraga nema nastavak sa pozitivnom masom. "
                "Pokušaj veću vrednost --sirina."
            )
        stanja = sorted(kandidati, key=lambda t: (-t[0], t[1]))[:sirina]

    kombinacija = stanja[0][1]
    if not (
        len(kombinacija) == len(set(kombinacija)) == 7
        and all(1 <= b <= 39 for b in kombinacija)
        and sum(kombinacija) == zbir
        and kombinacija[-1] - kombinacija[0] == raspon
    ):
        raise RuntimeError("NEXT nije prošao završnu proveru.")
    return kombinacija, zbir, raspon


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--koraci", type=int, default=3)
    parser.add_argument("--stabla", type=int, default=400)
    parser.add_argument("--sirina", type=int, default=64)
    args = parser.parse_args()

    if min(args.koraci, args.stabla, args.sirina) < 1:
        parser.error("Svi numerički parametri moraju biti pozitivni.")

    try:
        izvlacenja = ucitaj(args.csv)
        if len(izvlacenja) < args.koraci + 200:
            raise ValueError("Potrebno je najmanje 200 primera za učenje.")
        x, ciljevi, sledeci = pripremi(izvlacenja, args.koraci)
        modeli = obuci_modele(x, ciljevi, args.stabla)
        kombinacija, zbir, raspon = izaberi(
            modeli, sledeci, ostvarivi_profili(), args.sirina
        )
    except (OSError, ValueError, RuntimeError) as greska:
        parser.exit(1, f"Greška: {greska}\n")

    print("\nNEXT:", " ".join(f"{b:02d}" for b in kombinacija))
    print(f"Zbir: {zbir} | Raspon: {raspon}")


if __name__ == "__main__":
    main()



"""
Učenje distribucije zbira...
Učenje uslovne distribucije raspona...
Učenje strukture: broj 1/7...
Učenje strukture: broj 2/7...
Učenje strukture: broj 3/7...
Učenje strukture: broj 4/7...
Učenje strukture: broj 5/7...
Učenje strukture: broj 6/7...
Učenje strukture: broj 7/7...

NEXT: 04 x 12 y 27 z 37
Zbir: 143 | Raspon: 33
"""



"""
Učenje distribucije zbira...
Učenje uslovne distribucije raspona...
Učenje strukture: broj 1/7...
Učenje strukture: broj 2/7...
Učenje strukture: broj 3/7...
Učenje strukture: broj 4/7...
Učenje strukture: broj 5/7...
Učenje strukture: broj 6/7...
Učenje strukture: broj 7/7...

NEXT: 03 x 08 y 22 z 36
Zbir: 122 | Raspon: 33
"""



"""
Učenje distribucije zbira...
Učenje uslovne distribucije raspona...
Učenje strukture: broj 1/7...
Učenje strukture: broj 2/7...
Učenje strukture: broj 3/7...
Učenje strukture: broj 4/7...
Učenje strukture: broj 5/7...
Učenje strukture: broj 6/7...
Učenje strukture: broj 7/7...

NEXT: 04 x 15 y 30 z 37
Zbir: 152 | Raspon: 33
"""
