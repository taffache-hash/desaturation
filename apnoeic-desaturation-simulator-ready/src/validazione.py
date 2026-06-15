# -*- coding: utf-8 -*-
"""
================================================================================
BANCO DI VALIDAZIONE ESTERNA
Predizione forward del tempo-a-SpO2-soglia per popolazioni con parametri
congelati a priori, confronto con outcome clinici osservati.
================================================================================

NOTA SUI DATI OSSERVATI:
  I valori osservati (Sezione B) sono estratti VERBATIM dai PDF sorgente
  (Patel 1994; Xue 1996), riportati con media e SD come pubblicati. Il modello
  NON e' tarato su questi numeri: servono solo a misurare l'agreement a
  posteriori. Nessun input del modello e' derivato dagli outcome osservati.
"""

import numpy as np
import matplotlib.pyplot as plt
from apnea_desaturation_model import (
    Population, simula, tempo_a_soglia, metriche_agreement,
    frc_riferimento_ml, vo2_riferimento_ml_min,
    co_riferimento_dl_min, hb_riferimento_gdl, f_trapped_cc,
)

# =============================================================================
# SEZIONE A - POPOLAZIONI CON PARAMETRI CONGELATI A PRIORI
# =============================================================================
# COORTE DI VALIDAZIONE PRIMARIA: Patel R, Lenczyk M, Hannallah RS, McGill WA.
# "Age and the onset of desaturation in apnoeic children." Can J Anaesth
# 1994;41(9):771-4. (DOI 10.1007/BF03011582)
#   - 5 fasce d'eta', 10 pz/gruppo; preO2 fino a FetN2<3% (preO2 spinta);
#   - maschera rimossa, NESSUN jaw-thrust -> condizione VIE CHIUSE;
#   - outcome: tempo da rimozione maschera a SpO2 90%.
# Patel non pubblica i pesi medi per fascia: usiamo il peso mediano-per-eta'
# da riferimenti auxologici standard (WHO/CDC) all'eta' rappresentativa di
# ciascuna fascia. Questi pesi sono INDIPENDENTI dall'outcome (NON derivati da
# T90): nessuna covariata di input proviene dal risultato da predire.

# Coefficiente di atelettasia da assorbimento - CONGELATO da letteratura
# indipendente (Rothen 1995 / Hedenstierna: in anestesia con O2 100% lo shunt
# sale da ~5% a ~15% in pochi minuti). NON tarato su Patel/Xue.
DSHUNT_LETTERATURA = 0.10     # +10% -> shunt totale ~15%
TAU_ATEL = 90.0               # costante di tempo (s)

# Etichette inglesi delle fasce d'eta' per le legende delle figure (il paper
# e' in inglese; i nomi interni restano chiavi italiane per i dizionari).
ETA_EN = {"0-6 mesi": "0-6 mo", "7-23 mesi": "7-23 mo", "2-5 anni": "2-5 yr",
          "6-10 anni": "6-10 yr", "11-18 anni": "11-18 yr", "adulto sano": "adult"}

def costruisci(nome, eta, peso, fao2=0.90, airway_pervio=False, f_aw_o2=0.0,
               shunt=0.05, dshunt_max=0.0, use_cc=False):
    return Population(
        nome=nome, eta_anni=eta, peso_kg=peso,
        frc_ml=frc_riferimento_ml(eta, peso),
        vo2_ml_min=vo2_riferimento_ml_min(eta, peso),
        hb_gdl=hb_riferimento_gdl(eta),
        co_dl_min=co_riferimento_dl_min(eta, peso),
        shunt=shunt, fao2_iniz=fao2,
        airway_pervio=airway_pervio, f_aw_o2=f_aw_o2,
        dshunt_max=dshunt_max, tau_atel_s=TAU_ATEL,
        f_trapped=(f_trapped_cc(eta) if use_cc else 0.0),
    )

def build_patel(dshunt_max=0.0, use_cc=False):
    """Coorte Patel 1994 (5 fasce, vie chiuse, preO2 spinta)."""
    return [
        costruisci("0-6 mesi",   0.25,  6.0, dshunt_max=dshunt_max, use_cc=use_cc),
        costruisci("7-23 mesi",  1.25, 10.5, dshunt_max=dshunt_max, use_cc=use_cc),
        costruisci("2-5 anni",   3.5,  15.0, dshunt_max=dshunt_max, use_cc=use_cc),
        costruisci("6-10 anni",  8.0,  26.0, dshunt_max=dshunt_max, use_cc=use_cc),
        costruisci("11-18 anni", 14.5, 50.0, dshunt_max=dshunt_max, use_cc=use_cc),
    ]

def build_adulto(dshunt_max=0.0):
    return Population(nome="adulto sano", eta_anni=35, peso_kg=70,
                      frc_ml=2300, vo2_ml_min=250, hb_gdl=15, co_dl_min=50,
                      shunt=0.05, fao2_iniz=0.90, airway_pervio=False, f_aw_o2=0.0,
                      dshunt_max=dshunt_max, tau_atel_s=TAU_ATEL)

def build_thrive(dshunt_max=0.0):
    return Population(nome="adulto THRIVE", eta_anni=45, peso_kg=80,
                      frc_ml=2400, vo2_ml_min=250, hb_gdl=14, co_dl_min=55,
                      shunt=0.08, fao2_iniz=0.90, airway_pervio=True, f_aw_o2=1.0,
                      dshunt_max=dshunt_max, tau_atel_s=TAU_ATEL)

# =============================================================================
# SEZIONE B - OUTCOME OSSERVATI (estratti dai PDF sorgente, verbatim)
# =============================================================================
# Patel 1994 (CJA, DOI 10.1007/BF03011582), Tabella: T90 (s), media +/- SD,
# n=10/gruppo. Condizione: vie chiuse, preO2 a FetN2<3%.
OSSERVATI_90 = {
    "0-6 mesi":     96.5,   # +/- 12.7  (range 77-118)
    "7-23 mesi":   118.5,   # +/-  9.0  (range 79-163)
    "2-5 anni":    160.4,   # +/- 30.7  (range 114-205)
    "6-10 anni":   214.9,   # +/- 34.9  (range 165-274)
    "11-18 anni":  382.4,   # +/- 79.9  (range 185-490)
    "adulto sano": 364.0,   # riferimento citato da Patel (Jense 1991)
}
OSSERVATI_90_SD = {
    "0-6 mesi": 12.7, "7-23 mesi": 9.0, "2-5 anni": 30.7,
    "6-10 anni": 34.9, "11-18 anni": 79.9, "adulto sano": None,
}

# Secondo set indipendente: Xue 1996 (J Clin Anesth 8:568-574, 152 bambini),
# 3 fasce, preO2 2 min, vie chiuse. T90 (Subgruppo A) e T95 (medie).
# Pesi/Hb dalle demografiche del paper. Per validazione incrociata futura.
XUE_1996 = {
    # nome:        (eta, peso, Hb,   T99,   T95,   T90)
    "Xue 3mo-1a":  (0.8,  7.5, 12.4,  95.0, 110.2, 118.5),
    "Xue 1-3a":    (2.0, 12.6, 13.1, 127.5, 154.8, 168.7),
    "Xue 3-12a":   (7.6, 24.7, 13.8, 210.3, 231.8, 248.0),
}

# =============================================================================
# SEZIONE C - ESECUZIONE: PREDIZIONE FORWARD
# =============================================================================

def esegui_patel(cohort, etichetta=""):
    """Predizione forward su una coorte Patel; ritorna (risultati, curve, oss, pred)."""
    print(f"\n  -- Patel 1994  [{etichetta}] " + "-" * (40 - len(etichetta)))
    print(f"  {'gruppo':14s} {'pred':>7s} {'oss':>7s} {'err':>7s} {'err%':>6s}")
    risultati, curve, oss_l, pred_l = {}, {}, [], []
    for p in cohort:
        t, spo2, _ = simula(p, t_max_s=1800.0)
        t90 = tempo_a_soglia(t, spo2, 90.0)
        risultati[p.nome] = t90; curve[p.nome] = (t, spo2)
        oss = OSSERVATI_90.get(p.nome)
        if oss and t90:
            err = t90 - oss
            oss_l.append(oss); pred_l.append(t90)
            print(f"  {p.nome:14s} {t90:6.0f}s {oss:6.0f}s {err:+6.0f}s "
                  f"{100*err/oss:+5.0f}%")
    return risultati, curve, oss_l, pred_l

def riassunto_metriche(oss, pred, nome):
    m = metriche_agreement(oss, pred)
    mape = np.mean([abs(p-o)/o for o, p in zip(oss, pred)]) * 100
    r = np.corrcoef(oss, pred)[0, 1] if len(oss) > 1 else float("nan")
    print(f"  [{nome}] n={m['n']} | bias {m['bias_s']:+.0f}s | RMSE {m['rmse_s']:.0f}s "
          f"| MAPE {mape:.0f}% | r={r:.3f}")
    return mape, m['bias_s'], r

# =============================================================================
# SEZIONE D - GRAFICI
# =============================================================================

def grafico_confronto(curve_base, curve_din, ob, pb, od, pd, oxb, pxb, oxd, pxd):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    colori = {"0-6 mesi": "C0", "7-23 mesi": "C1", "2-5 anni": "C2",
              "6-10 anni": "C3", "11-18 anni": "C4"}
    for nome, c in colori.items():
        tb, sb = curve_base[nome]; td, sd = curve_din[nome]
        ax1.plot(tb / 60.0, sb, ls="--", lw=1.3, color=c, alpha=0.55)
        ax1.plot(td / 60.0, sd, ls="-", lw=2.0, color=c, label=ETA_EN.get(nome, nome))
        oss = OSSERVATI_90.get(nome); sd_ = OSSERVATI_90_SD.get(nome)
        if oss:
            ax1.errorbar(oss/60.0, 90, xerr=(sd_/60.0 if sd_ else None),
                         fmt="o", color=c, ms=7, capsize=3, mec="k", mew=0.7)
    ax1.axhline(90, ls=":", c="gray", lw=1)
    ax1.set_xlim(0, 8); ax1.set_ylim(60, 101)
    ax1.set_xlabel("Apnoea time (min)"); ax1.set_ylabel("SpO$_2$ (%)")
    ax1.set_title("Patel: baseline (- -) vs closing capacity (—) + obs. ($\\bullet$)")
    ax1.legend(fontsize=8, loc="lower left"); ax1.grid(alpha=0.3)

    lim = 520
    ax2.plot([0, lim], [0, lim], "k--", lw=1, label="identity")
    ax2.scatter(ob, pb, marker="o", facecolor="none", edgecolor="C3", s=70,
                label="Patel baseline")
    ax2.scatter(od, pd, marker="o", color="C3", s=70, edgecolor="k",
                label="Patel closing cap.")
    ax2.scatter(oxb, pxb, marker="s", facecolor="none", edgecolor="C0", s=70,
                label="Xue baseline")
    ax2.scatter(oxd, pxd, marker="s", color="C0", s=70, edgecolor="k",
                label="Xue closing cap.")
    ax2.set_xlim(0, lim); ax2.set_ylim(0, lim)
    ax2.set_xlabel("Observed T$_{90}$ (s)"); ax2.set_ylabel("Predicted T$_{90}$ (s)")
    ax2.set_title("Agreement: the module moves points toward identity")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3); ax2.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("desaturazione_cc.png", dpi=130)
    print("\n  Figura salvata: desaturazione_cc.png")

def cross_validazione_xue(dshunt_max=0.0, etichetta="", use_cc=False):
    """Validazione incrociata fuori campione: STESSO modello, coorte
    indipendente (Xue 1996), protocollo diverso (preO2 2 min + suxametonio)."""
    print(f"\n  -- Xue 1996  [{etichetta}] " + "-" * (42 - len(etichetta)))
    print(f"  {'gruppo':14s} {'pred':>7s} {'oss':>7s} {'err':>7s} {'err%':>6s}")
    oss_list, pred_list = [], []
    for nome, (eta, peso, hb, t99, t95, t90) in XUE_1996.items():
        p = Population(
            nome=nome, eta_anni=eta, peso_kg=peso,
            frc_ml=frc_riferimento_ml(eta, peso),
            vo2_ml_min=vo2_riferimento_ml_min(eta, peso),
            hb_gdl=hb, co_dl_min=co_riferimento_dl_min(eta, peso),
            shunt=0.05, fao2_iniz=0.90, airway_pervio=False, f_aw_o2=0.0,
            dshunt_max=dshunt_max, tau_atel_s=TAU_ATEL,
            f_trapped=(f_trapped_cc(eta) if use_cc else 0.0))
        t, spo2, _ = simula(p, t_max_s=1800.0)
        pred = tempo_a_soglia(t, spo2, 90.0)
        err = pred - t90
        oss_list.append(t90); pred_list.append(pred)
        print(f"  {nome:14s} {pred:6.0f}s {t90:6.0f}s {err:+6.0f}s "
              f"{100*err/t90:+5.0f}%")
    return oss_list, pred_list

def grafico_curve():
    """Figura esplicativa (in inglese): gradiente eta'-dipendente (vie chiuse)
    e caso limite ossigenazione apneica (THRIVE) vs vie chiuse.
    Rigenera desaturazione_curve.png coerente col modello corrente."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    colori = ["C0", "C1", "C2", "C3", "C4", "C5"]
    pops = build_patel(0.0, use_cc=False) + [build_adulto(0.0)]
    for p, c in zip(pops, colori):
        t, spo2, _ = simula(p, t_max_s=1800.0)
        etich = ETA_EN.get(p.nome, p.nome)
        ax1.plot(t / 60.0, spo2, lw=2.0, color=c, label=etich)
    ax1.axhline(90, ls="--", c="gray", lw=1)
    ax1.set_xlim(0, 12); ax1.set_ylim(60, 101)
    ax1.set_xlabel("Apnoea time (min)"); ax1.set_ylabel("SpO$_2$ (%)")
    ax1.set_title("Age-dependent gradient (closed airway)")
    ax1.legend(fontsize=8, loc="lower left"); ax1.grid(alpha=0.3)

    tA, sA, _ = simula(build_adulto(0.0), t_max_s=1800.0)
    tT, sT, _ = simula(build_thrive(0.0), t_max_s=1800.0)
    ax2.plot(tA / 60.0, sA, lw=2.2, color="C3", label="closed airway (adult)")
    ax2.plot(tT / 60.0, sT, lw=2.2, color="C2", label="apnoeic oxygenation (THRIVE)")
    ax2.axhline(90, ls="--", c="gray", lw=1)
    ax2.set_xlim(0, 20); ax2.set_ylim(60, 101)
    ax2.set_xlabel("Apnoea time (min)"); ax2.set_ylabel("SpO$_2$ (%)")
    ax2.set_title("Limiting case: apnoeic oxygenation (THRIVE) vs closed airway")
    ax2.legend(fontsize=8, loc="lower left"); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("desaturazione_curve.png", dpi=130)
    print("  Figura salvata: desaturazione_curve.png")

def grafico_shunt():
    """Figura SUPPLEMENTARE (inglese): ipotesi shunt dinamico RESPINTA.
    Shunt fisso vs shunt dinamico (atelettasia da assorbimento): le curve si
    sovrappongono -> nessun effetto sull'onset della desaturazione.
    Salva desaturazione_shunt_suppl.png (figura di supplemento, non principale)."""
    _, cur_fix, of, pf = esegui_patel(build_patel(0.0, use_cc=False), "shunt fixed")
    _, cur_dyn, od, pd = esegui_patel(build_patel(DSHUNT_LETTERATURA, use_cc=False),
                                      "shunt dynamic")
    oxf, pxf = cross_validazione_xue(0.0, "shunt fixed")
    oxd, pxd = cross_validazione_xue(DSHUNT_LETTERATURA, "shunt dynamic")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    colori = {"0-6 mesi": "C0", "7-23 mesi": "C1", "2-5 anni": "C2",
              "6-10 anni": "C3", "11-18 anni": "C4"}
    for nome, c in colori.items():
        tf, sf = cur_fix[nome]; td, sd = cur_dyn[nome]
        ax1.plot(tf / 60.0, sf, ls="--", lw=1.3, color=c, alpha=0.55)
        ax1.plot(td / 60.0, sd, ls="-", lw=2.0, color=c, label=ETA_EN.get(nome, nome))
        oss = OSSERVATI_90.get(nome); sd_ = OSSERVATI_90_SD.get(nome)
        if oss:
            ax1.errorbar(oss/60.0, 90, xerr=(sd_/60.0 if sd_ else None),
                         fmt="o", color=c, ms=7, capsize=3, mec="k", mew=0.7)
    ax1.axhline(90, ls=":", c="gray", lw=1)
    ax1.set_xlim(0, 8); ax1.set_ylim(60, 101)
    ax1.set_xlabel("Apnoea time (min)"); ax1.set_ylabel("SpO$_2$ (%)")
    ax1.set_title("Suppl.: fixed shunt (- -) vs dynamic shunt (—) + obs. ($\\bullet$)")
    ax1.legend(fontsize=8, loc="lower left"); ax1.grid(alpha=0.3)

    lim = 520
    ax2.plot([0, lim], [0, lim], "k--", lw=1, label="identity")
    ax2.scatter(of, pf, marker="o", facecolor="none", edgecolor="C3", s=70,
                label="Patel fixed shunt")
    ax2.scatter(od, pd, marker="o", color="C3", s=70, edgecolor="k",
                label="Patel dynamic shunt")
    ax2.scatter(oxf, pxf, marker="s", facecolor="none", edgecolor="C0", s=70,
                label="Xue fixed shunt")
    ax2.scatter(oxd, pxd, marker="s", color="C0", s=70, edgecolor="k",
                label="Xue dynamic shunt")
    ax2.set_xlim(0, lim); ax2.set_ylim(0, lim)
    ax2.set_xlabel("Observed T$_{90}$ (s)"); ax2.set_ylabel("Predicted T$_{90}$ (s)")
    ax2.set_title("Agreement unchanged: rising shunt does not shift onset")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3); ax2.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("desaturazione_shunt_suppl.png", dpi=130)
    print("  Figura salvata: desaturazione_shunt_suppl.png")

if __name__ == "__main__":
    print("=" * 74)
    print("CONFRONTO SCENARI: baseline vs MODULO CLOSING CAPACITY")
    print(f"Trapping da letteratura (Mansell 1972): f0={f_trapped_cc(0):.2f} (lattante)"
          f" -> {f_trapped_cc(14.5)*100:.0f}% (adolescente). NON tarato sui trial.")
    print("=" * 74)

    # frazioni intrappolate per eta' (trasparenza)
    print("  frazione FRC intrappolata per fascia (closing capacity):")
    for nome, eta in [("0-6 mesi",0.25),("7-23 mesi",1.25),("2-5 anni",3.5),
                      ("6-10 anni",8.0),("11-18 anni",14.5)]:
        print(f"    {nome:12s} eta {eta:4.1f}a -> f_trapped = {f_trapped_cc(eta)*100:4.1f}%")

    # --- scenario A: baseline (no closing capacity) ---
    res_b, cur_b, ob, pb = esegui_patel(build_patel(0.0, use_cc=False), "baseline")
    oxb, pxb = cross_validazione_xue(0.0, "baseline", use_cc=False)

    # --- scenario B: closing capacity ON (coefficiente di letteratura) ---
    res_c, cur_c, oc, pc = esegui_patel(build_patel(0.0, use_cc=True), "closing capacity")
    oxc, pxc = cross_validazione_xue(0.0, "closing capacity", use_cc=True)

    print("\n" + "=" * 74)
    print("METRICHE DI AGREEMENT")
    print("=" * 74)
    riassunto_metriche(ob, pb, "Patel  baseline        ")
    riassunto_metriche(oc, pc, "Patel  closing capacity")
    riassunto_metriche(oxb, pxb, "Xue    baseline        ")
    riassunto_metriche(oxc, pxc, "Xue    closing capacity")

    tT, sT, _ = simula(build_thrive(0.0), t_max_s=1800.0)
    tt90 = tempo_a_soglia(tT, sT, 90.0)
    print(f"\n  Controllo THRIVE: t->90% = "
          f"{'nessuna desat' if tt90 is None else f'{tt90:.0f}s'}")

    grafico_confronto(cur_b, cur_c, ob, pb, oc, pc, oxb, pxb, oxc, pxc)
    grafico_curve()
    grafico_shunt()
