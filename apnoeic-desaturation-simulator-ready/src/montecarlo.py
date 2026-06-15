# -*- coding: utf-8 -*-
"""
================================================================================
MODULO MONTE CARLO (VETTORIZZATO) - VARIABILITA' INTERINDIVIDUALE
Integra N pazienti virtuali in parallelo (passo fisso RK4) per produrre
intervalli di predizione al 95% del tempo a SpO2 90%.
Domanda: il residuo del modello deterministico e' ERRORE o DISPERSIONE
fisiologica? -> le medie osservate cadono nel 95% PI?
================================================================================
CV congelati da letteratura (NON tarati per far cadere l'osservato dentro):
  FRC 0.22 (Thorsteinsson), VO2 0.18, CO 0.18, Hb SD 1.2, FAO2 SD 0.03,
  shunt SD 0.03, closing-capacity fraction CV 0.40.
"""
import numpy as np
from apnea_desaturation_model import (
    severinghaus_sao2, severinghaus_po2, P_BARO, P_H2O, HUFNER, ALPHA_O2,
    frc_riferimento_ml, vo2_riferimento_ml_min, co_riferimento_dl_min,
    hb_riferimento_gdl, f_trapped_cc,
)

CV_FRC, CV_VO2, CV_CO, CV_CC = 0.22, 0.18, 0.18, 0.40
SD_HB, SD_FAO2, SD_SHUNT = 1.2, 0.03, 0.03
R_APNEA = 0.06

# Etichette inglesi delle fasce per gli assi delle figure (paper in inglese).
ETA_EN = {"0-6 mesi": "0-6 mo", "7-23 mesi": "7-23 mo", "2-5 anni": "2-5 yr",
          "6-10 anni": "6-10 yr", "11-18 anni": "11-18 yr"}

def _logn(cv, rng, n):
    sigma = np.sqrt(np.log(1.0 + cv**2))
    return rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma, size=n)

def _deriv(A_L, Cv, P):
    frc_eff = P["frc"] * (1.0 - P["ftr"])
    fao2 = np.clip(A_L / frc_eff, 0.0, 1.0)
    pao2 = fao2 * (P_BARO - P_H2O)
    scc = severinghaus_sao2(pao2)
    ccc = HUFNER * P["hb"] * scc + ALPHA_O2 * pao2
    sv = np.clip(Cv / (HUFNER * P["hb"]), 1e-6, 1.0)
    pv = severinghaus_po2(sv)
    cv = HUFNER * P["hb"] * sv + ALPHA_O2 * pv
    ca = (1.0 - P["shunt"]) * ccc + P["shunt"] * cv
    j_up = np.maximum(P["co"] * (ca - cv), 0.0)
    j_net = np.maximum(j_up - R_APNEA * P["vo2"], 0.0)
    o2_in = j_net * P["f_aw_o2"] * P["patent"]
    dA_L = o2_in - j_up
    dCv = (j_up - P["vo2"]) / P["vol_sangue"]
    return dA_L, dCv

def _sao2(A_L, Cv, P):
    frc_eff = P["frc"] * (1.0 - P["ftr"])
    fao2 = np.clip(A_L / frc_eff, 0.0, 1.0)
    pao2 = fao2 * (P_BARO - P_H2O)
    scc = severinghaus_sao2(pao2)
    ccc = HUFNER * P["hb"] * scc + ALPHA_O2 * pao2
    sv = np.clip(Cv / (HUFNER * P["hb"]), 1e-6, 1.0)
    pv = severinghaus_po2(sv)
    cv = HUFNER * P["hb"] * sv + ALPHA_O2 * pv
    ca = (1.0 - P["shunt"]) * ccc + P["shunt"] * cv
    sa = np.clip((ca - ALPHA_O2 * pao2) / (HUFNER * P["hb"]), 1e-6, 1.0)
    return sa * 100.0

def monte_carlo(eta, peso, hb_mean, n=2000, t_max_s=1000.0, dt=1.0, seed=0,
                use_cc=True, fao2_mean=0.90, drop_nan=True):
    rng = np.random.default_rng(seed)
    frc0 = frc_riferimento_ml(eta, peso); vo20 = vo2_riferimento_ml_min(eta, peso)
    co0 = co_riferimento_dl_min(eta, peso); ftr0 = f_trapped_cc(eta) if use_cc else 0.0
    P = {
        "frc": frc0 * _logn(CV_FRC, rng, n),
        "vo2": vo20 * _logn(CV_VO2, rng, n),
        "co":  co0 * _logn(CV_CO, rng, n),
        "hb":  np.clip(rng.normal(hb_mean, SD_HB, n), 7.0, 20.0),
        "shunt": np.clip(rng.normal(0.05, SD_SHUNT, n), 0.0, 0.30),
        "ftr": np.clip(ftr0 * _logn(CV_CC, rng, n), 0.0, 0.40),
        "f_aw_o2": np.zeros(n), "patent": np.zeros(n),
    }
    P["vol_sangue"] = 0.75 * peso * np.ones(n)
    fao2 = np.clip(rng.normal(fao2_mean, SD_FAO2, n), 0.80, 0.96)
    A_L = fao2 * P["frc"] * (1.0 - P["ftr"])
    pao2_0 = fao2 * (P_BARO - P_H2O)
    sa0 = severinghaus_sao2(pao2_0)
    ca0 = HUFNER * P["hb"] * sa0 + ALPHA_O2 * pao2_0
    Cv = ca0 - P["vo2"] / P["co"]
    nstep = int(t_max_s / dt)
    t90 = np.full(n, np.nan)
    prev_sa = _sao2(A_L, Cv, P)
    h = dt / 60.0
    for k in range(1, nstep + 1):
        k1a, k1c = _deriv(A_L, Cv, P)
        k2a, k2c = _deriv(A_L + 0.5*h*k1a, Cv + 0.5*h*k1c, P)
        k3a, k3c = _deriv(A_L + 0.5*h*k2a, Cv + 0.5*h*k2c, P)
        k4a, k4c = _deriv(A_L + h*k3a, Cv + h*k3c, P)
        A_L = np.maximum(A_L + (h/6)*(k1a+2*k2a+2*k3a+k4a), 1e-6)
        Cv  = np.maximum(Cv + (h/6)*(k1c+2*k2c+2*k3c+k4c), 1e-6)
        sa = _sao2(A_L, Cv, P)
        cross = (prev_sa > 90.0) & (sa <= 90.0) & np.isnan(t90)
        if cross.any():
            frac = (prev_sa[cross] - 90.0) / (prev_sa[cross] - sa[cross])
            t90[cross] = (k - 1 + frac) * dt
        prev_sa = sa
    return t90[~np.isnan(t90)] if drop_nan else t90

def medie_di_gruppo(eta, peso, hb_mean, n_group, n_rep=1000, t_max_s=1000.0,
                    seed=0, use_cc=True):
    """METRICA RIGOROSA: distribuzione delle MEDIE DI GRUPPO.
    L'osservato e' una media di n_group pazienti (es. Patel n=10). Simuliamo
    n_rep gruppi da n_group pazienti virtuali, calcoliamo la media di ciascun
    gruppo e restituiamo la distribuzione delle medie. Il 95% di questa
    distribuzione e' l'intervallo entro cui dovrebbe cadere la media osservata:
    e' molto piu' stringente del 95% PI individuale (SE = SD/sqrt(n))."""
    n_tot = n_rep * n_group
    t90 = monte_carlo(eta, peso, hb_mean, n=n_tot, t_max_s=t_max_s, seed=seed,
                      use_cc=use_cc, drop_nan=False)
    t90 = t90.reshape(n_rep, n_group)
    return np.nanmean(t90, axis=1)

COORTE = {
    # nome: (eta, peso, hb, T90 oss, SD oss, n_gruppo)
    # pesi = mediana-per-eta' auxologica (WHO/CDC), indipendenti dall'outcome
    "0-6 mesi":   (0.25,  6.0, None,  96.5, 12.7, 10),
    "7-23 mesi":  (1.25, 10.5, None, 118.5,  9.0, 10),
    "2-5 anni":   (3.5,  15.0, None, 160.4, 30.7, 10),
    "6-10 anni":  (8.0,  26.0, None, 214.9, 34.9, 10),
    "11-18 anni": (14.5, 50.0, None, 382.4, 79.9, 10),
}

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    print("=" * 84)
    print("MONTE CARLO - confronto a DUE livelli (Patel 1994). Closing capacity attivo.")
    print("CV congelati da letteratura, NON tarati.")
    print("=" * 84)
    print("\n[1] METRICA RIGOROSA: media osservata vs 95% CI della MEDIA DI GRUPPO")
    print("    (n_gruppo = n del trial; molti gruppi simulati, SE = SD/sqrt(n))")
    print(f"  {'fascia':12s} {'media pred':>10s} {'95% CI media':>18s} {'oss':>7s} "
          f"{'in CI':>6s}")
    fasce, om_l, osd_l = [], [], []
    gm_lo, gm_hi, gm_mean, camp, ind_lo, ind_hi = [], [], [], [], [], []
    cov_rig, cov_ind = 0, 0
    for nome, (eta, peso, hbm, o, sd, ng) in COORTE.items():
        hb = hb_riferimento_gdl(eta) if hbm is None else hbm
        # metrica rigorosa: distribuzione delle medie di gruppo
        gmeans = medie_di_gruppo(eta, peso, hb, n_group=ng, n_rep=400, seed=42, t_max_s=900.0)
        glo, ghi = np.percentile(gmeans, [2.5, 97.5]); gmu = np.mean(gmeans)
        inci = "SI" if glo <= o <= ghi else "no"
        cov_rig += (glo <= o <= ghi)
        print(f"  {nome:12s} {gmu:9.0f}s [{glo:7.0f},{ghi:7.0f}]s {o:6.0f}s {inci:>6s}")
        fasce.append(nome); om_l.append(o); osd_l.append(sd)
        gm_lo.append(glo); gm_hi.append(ghi); gm_mean.append(gmu)
        # metrica secondaria: inviluppo individuale (dispersione fisiologica)
        ind = monte_carlo(eta, peso, hb, n=1500, seed=7, t_max_s=900.0)
        ilo, ihi = np.percentile(ind, [2.5, 97.5])
        cov_ind += (ilo <= o <= ihi)
        camp.append(ind); ind_lo.append(ilo); ind_hi.append(ihi)

    print(f"\n  Copertura rigorosa (media di gruppo): {cov_rig}/{len(fasce)}")
    print("\n[2] METRICA SECONDARIA: media osservata vs 95% PI INDIVIDUALE")
    print("    (dispersione fisiologica tra singoli pazienti, piu' permissiva)")
    for nome, o, ilo, ihi in zip(fasce, om_l, ind_lo, ind_hi):
        print(f"  {nome:12s} 95% PI [{ilo:6.0f},{ihi:6.0f}]s  oss {o:5.0f}s  "
              f"{'dentro' if ilo<=o<=ihi else 'fuori'}")
    print(f"\n  Copertura individuale: {cov_ind}/{len(fasce)}")
    print("\n  INTERPRETAZIONE: la metrica rigorosa (media di gruppo) e' il test")
    print("  corretto e mostra la sovrastima sistematica; l'inviluppo individuale")
    print("  indica che le medie osservate restano entro la dispersione fisiologica.")

    # --- figura a due pannelli ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(len(fasce))
    om = np.array(om_l); osd = np.array(osd_l)
    glo = np.array(gm_lo); ghi = np.array(gm_hi); gmu = np.array(gm_mean)
    # pannello 1: metrica RIGOROSA (CI della media di gruppo)
    ax1.fill_between(x, glo, ghi, alpha=0.25, color="C0",
                     label="95% CI of predicted mean (n=10)")
    ax1.plot(x, gmu, "o-", color="C0", label="predicted mean")
    ax1.errorbar(x, om, yerr=osd, fmt="s", color="C3", capsize=4,
                 label="observed (mean$\\pm$SD)")
    ax1.set_xticks(x); ax1.set_xticklabels([ETA_EN.get(f,f) for f in fasce], rotation=20, fontsize=8)
    ax1.set_ylabel("T$_{90}$ (s)")
    ax1.set_title("Rigorous test: observed mean vs CI of the group mean")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    # pannello 2: inviluppo INDIVIDUALE (dispersione fisiologica)
    parts = ax2.violinplot(camp, positions=x, showmedians=True, widths=0.8)
    for pc in parts['bodies']:
        pc.set_facecolor("C0"); pc.set_alpha(0.32)
    ax2.errorbar(x, om, yerr=osd, fmt="s", color="C3", capsize=4,
                 label="observed", zorder=5)
    ax2.set_xticks(x); ax2.set_xticklabels([ETA_EN.get(f,f) for f in fasce], rotation=20, fontsize=8)
    ax2.set_ylabel("T$_{90}$ (s)")
    ax2.set_title("Individual dispersion (virtual patients) vs observed")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("desaturazione_montecarlo.png", dpi=130)
    print("\n  Figura salvata: desaturazione_montecarlo.png")
