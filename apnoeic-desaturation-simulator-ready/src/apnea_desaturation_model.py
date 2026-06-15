# -*- coding: utf-8 -*-
"""
================================================================================
MODELLO IN-SILICO DI DESATURAZIONE IN APNEA
Validazione esterna prospettica di un modello matematico deterministico
contro dati clinici reali (Xue 1996, THRIVE/Patel 2015, Benumof 1997)
================================================================================

IMPIANTO METODOLOGICO (mutuato dal lavoro DPSV):
  - Modello MECCANICISTICO a compartimenti (ODE), NON un learner statistico.
  - Parametri CONGELATI A PRIORI da fonti indipendenti (equazioni di
    riferimento per FRC, VO2, Hb, gittata) -> NESSUN fitting sui trial.
  - Predizione FORWARD del tempo-a-SpO2-soglia per ciascuna popolazione.
  - Confronto con outcome clinici osservati tramite metriche di agreement
    pre-specificate (bias, RMSE, Bland-Altman, copertura).

ARCHITETTURA FISIOLOGICA:
  Compartimento 1 - Gas alveolare (riserva polmonare di O2, ~ FRC)
  Compartimento 2 - Riserva ematica+tissutale di O2 (sangue + mioglobina)
  Accoppiamento   - Curva di dissociazione Hb-O2 (Severinghaus),
                    equazione dello shunt (venous admixture),
                    principio di Fick (consumo metabolico VO2),
                    FLUSSO DI MASSA APNEICO (chiave dell'ossigenazione apneica).

Il meccanismo decisivo e' il flusso di massa apneico: durante l'apnea l'O2
e' rimosso dall'alveolo piu' velocemente di quanto la CO2 vi sia aggiunta,
generando un gradiente che richiama gas dal faringe. Se le vie aeree sono
pervie e riempite di O2 (THRIVE), questo gas e' O2 puro e l'O2 alveolare
resta in plateau; se le vie sono chiuse, non c'e' rimpiazzo e l'O2 alveolare
crolla. La CONDIZIONE SPERIMENTALE PRIMARIA e' codificata dalla pervieta' delle
vie aeree e dalla frazione di O2 del gas inspirato (F_aw_O2); moduli secondari
di sensibilita' (atelettasia da assorbimento e closing capacity eta'-dipendente)
sono estensioni meccanicistiche opzionali, disattivate di default, esplorate in
analisi di sensibilita' e NON parte del modello di base.
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field

# =============================================================================
# SEZIONE 1 - COSTANTI FISIOLOGICHE E CURVA DI DISSOCIAZIONE Hb-O2
# =============================================================================

P_BARO   = 760.0   # pressione barometrica (mmHg, livello del mare)
P_H2O    = 47.0    # tensione vapore acqueo a 37 C (mmHg)
HUFNER   = 1.34    # capacita' di legame O2 dell'Hb (mL O2 / g Hb)
ALPHA_O2 = 0.003   # solubilita' O2 plasma (mL O2 / dL / mmHg)

def severinghaus_sao2(po2_mmHg):
    """Curva di dissociazione Hb-O2 (Severinghaus 1979). PO2 -> SaO2 [frazione].
    Forma invertibile, valida 10-500 mmHg."""
    po2 = np.maximum(po2_mmHg, 1e-6)
    return 1.0 / ((23400.0 / (po2**3 + 150.0 * po2)) + 1.0)

def severinghaus_po2(sao2_frac):
    """Inversa numerica SaO2 -> PO2 (bisezione). Scalare->scalare, array->array."""
    scalar_in = np.isscalar(sao2_frac) or (np.ndim(sao2_frac) == 0)
    sao2 = np.clip(np.atleast_1d(sao2_frac).astype(float), 1e-6, 0.999999)
    lo = np.full_like(sao2, 1.0)
    hi = np.full_like(sao2, 700.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f = severinghaus_sao2(mid)
        hi = np.where(f > sao2, mid, hi)
        lo = np.where(f > sao2, lo, mid)
    out = 0.5 * (lo + hi)
    return float(out[0]) if scalar_in else out

def o2_content(hb_gdl, sao2_frac, po2_mmHg):
    """Contenuto di O2 nel sangue (mL O2 / dL)."""
    return HUFNER * hb_gdl * sao2_frac + ALPHA_O2 * po2_mmHg

# =============================================================================
# SEZIONE 2 - PARAMETRI CONGELATI A PRIORI (per popolazione/condizione)
# =============================================================================

@dataclass
class Population:
    """Set di parametri fisiologici fissati A PRIORI da fonti indipendenti.
    Nessun parametro qui dentro e' tarato sui trial di validazione."""
    nome: str
    eta_anni: float          # eta'
    peso_kg: float           # peso corporeo
    frc_ml: float            # capacita' funzionale residua (riserva polmonare)
    vo2_ml_min: float        # consumo di O2 metabolico
    hb_gdl: float            # emoglobina
    co_dl_min: float         # gittata cardiaca (dL/min)
    shunt: float             # frazione di shunt Qs/Qt (venous admixture)
    fao2_iniz: float         # frazione O2 alveolare iniziale (qualita' preO2)
    # --- condizione delle vie aeree durante l'apnea ---
    airway_pervio: bool      # True = gas entra dal faringe; False = vie chiuse
    f_aw_o2: float           # frazione O2 del gas delle vie aeree (1.0=THRIVE, 0.21=aria)
    r_apnea: float = 0.06    # quoziente respiratorio apneico (CO2 alv / O2 consumato)
    vol_sangue_dl: float = None  # volume ematico effettivo (riserva); se None -> stima
    # --- MODULO SHUNT DINAMICO (atelettasia da assorbimento) ---
    # shunt(t) = shunt + dshunt_max*(1-exp(-t/tau))*[fattore vie aeree]
    # Coefficienti da letteratura INDIPENDENTE (Rothen/Hedenstierna: in anestesia
    # con O2 100% lo shunt sale da ~5% a ~15% in pochi minuti). dshunt_max=0 -> modulo OFF.
    dshunt_max: float = 0.0      # incremento massimo di shunt (es. 0.10 -> 5%+10%=15%)
    tau_atel_s: float = 90.0     # costante di tempo dell'atelettasia (s)
    atel_patent_factor: float = 0.3  # vie pervie+O2 splintano -> atelettasia ridotta
    f_trapped: float = 0.0       # frazione di FRC intrappolata (closing capacity); 0 -> OFF

    def __post_init__(self):
        if self.vol_sangue_dl is None:
            # volemia ~ 70-80 mL/kg; il "serbatoio" O2 include anche tessuti
            self.vol_sangue_dl = 0.75 * self.peso_kg  # 75 mL/kg -> dL = *0.1*... vedi nota
            # 75 mL/kg = 0.75 dL/kg  -> peso_kg * 0.75 dL  (corretto)

# --- Equazioni di riferimento pediatriche (NON tarate sui trial) ----------
# Ogni coefficiente e' ancorato a una fonte indipendente. Le leggi sono
# CONTINUE (no bande a gradino): cosi' il rapporto riserva/consumo FRC/VO2
# varia con il peso in modo fisiologico e il gradiente eta'-dipendente emerge
# spontaneamente invece di essere imposto.

def vo2_riferimento_ml_min(eta_anni, peso_kg):
    """Consumo metabolico di O2 via scaling allometrico (Kleiber/West):
        VO2 = k * peso^0.75
    Riproduce intrinsecamente il VO2/kg piu' alto nei piccoli (driver primario
    della desaturazione rapida) senza bande arbitrarie.
    Calibrazione: adulto 70 kg -> 250 mL/min (3.5 mL/kg/min).
    Verifica: neonato 3.5 kg -> ~7.6 mL/kg/min (coerente con 6-8 atteso).
    [rif.] scaling metabolico allometrico; VO2 neonatale di riferimento."""
    K_VO2 = 250.0 / (70.0 ** 0.75)              # ~10.33
    return K_VO2 * peso_kg ** 0.75

def frc_riferimento_ml(eta_anni, peso_kg):
    """Capacita' funzionale residua EFFETTIVA in ANESTESIA/supino/paralisi
    (riserva polmonare di O2). Valori da Thorsteinsson 1990 (FRC misurata in
    bambini anestetizzati), citato da tutti i trial bersaglio -> fonte
    INDIPENDENTE dagli outcome, quindi congelata a priori:
        FRC ~ 17 mL/kg nel lattante (<1 a), ~24 mL/kg oltre l'anno,
        asintotica a ~30 mL/kg nell'adulto anestetizzato.
    Funzione liscia monotona:  k_frc(eta) = 30 - 13*exp(-eta/2.5)  [mL/kg]
        eta 0 -> 17.0 ; eta 2 -> 24.2 ; eta 5 -> 28.2 ; adulto -> ~30.
    NB: la FRC in anestesia e' MOLTO minore di quella da sveglio: usarla e'
    decisivo per non sovrastimare il tempo di apnea, specie nei piccoli."""
    k_frc = 30.0 - 13.0 * np.exp(-eta_anni / 2.5)
    return k_frc * peso_kg

def co_riferimento_dl_min(eta_anni, peso_kg):
    """Gittata cardiaca via scaling allometrico: CO = k * peso^0.75.
    Calibrazione: adulto 70 kg -> 5 L/min (50 dL/min).
    CO/kg piu' alta nei piccoli (neonato ~150 mL/kg/min), coerente.
    [rif.] scaling allometrico della gittata."""
    K_CO = 5000.0 / (70.0 ** 0.75)              # mL/min, ~206.6
    return (K_CO * peso_kg ** 0.75) / 100.0     # -> dL/min

def hb_riferimento_gdl(eta_anni):
    """Emoglobina di riferimento per eta' (curva fisiologica con nadir a ~2-3
    mesi). Influenza la riserva ematica di O2 (contenuto proporzionale a Hb).
    NB CRITICO: <3 mesi prevale l'Hb fetale con curva di dissociazione
    SINISTRORSA (P50 ~19 vs 26.8 mmHg): NON modellata qui -> vedi LIMITI.
    [rif.] valori di Hb per eta'."""
    eta_nodi = [0.0, 0.05, 0.25, 0.5, 1.0, 5.0, 12.0, 18.0]
    hb_nodi  = [17.0, 15.0, 11.0, 11.5, 12.0, 12.5, 13.5, 14.5]
    return float(np.interp(eta_anni, eta_nodi, hb_nodi))

# --- MODULO CLOSING CAPACITY (ESTENSIONE DI SENSIBILITA', non modello base) ---
# Estensione meccanicistica OPZIONALE (default OFF, f_trapped=0). NON fa parte
# del modello principale congelato a priori: e' un modulo esplorato in analisi
# di sensibilita' per quantificare quanto la riduzione della FRC effettiva da
# closing capacity riduca la sovrastima sistematica dei tempi.
# Coefficienti ancorati a letteratura indipendente (Mansell 1972; CC/FRC alta
# nei piccoli, trascurabile dall'eta' scolare; in anestesia la FRC scende sotto
# la CC nei giovani). f0 = picco di trapping nel lattante: e' il parametro PIU'
# INCERTO (estrapolazione teorica sotto i 6 anni), riportato come banda di
# sensibilita' e NON tarato sui trial.
F0_CC = 0.22          # frazione di FRC dietro vie chiuse nel lattante (~22%)
AGE_SCALE_CC = 4.0    # decadimento (anni): ~6% a 12 a, ~0 in adolescenza

def f_trapped_cc(eta_anni, f0=F0_CC, scale=AGE_SCALE_CC):
    """Frazione di FRC sotto il closing volume (intrappolata, non scambiante e
    mal denitrogenata) in funzione dell'eta'. Forma esponenziale decrescente:
        f(eta) = f0 * exp(-eta/scale)
    Ancorata a: trapping alto nel lattante, ~nullo dall'adolescenza (gli 11-18
    erano gia' ben predetti -> verifica di coerenza, non fitting)."""
    return f0 * np.exp(-eta_anni / scale)

# =============================================================================
# SEZIONE 3 - MODELLO ODE (CORE)
# =============================================================================

def shunt_at(t_s, p: Population):
    """Frazione di shunt al tempo t (s). Cresce per atelettasia da assorbimento
    durante l'apnea; attenuata se le vie sono pervie con O2 (effetto splinting).
    Riduce a shunt costante se p.dshunt_max == 0. Gestisce scalari e array."""
    factor = p.atel_patent_factor if p.airway_pervio else 1.0
    return p.shunt + p.dshunt_max * (1.0 - np.exp(-t_s / p.tau_atel_s)) * factor

def apnea_odes(t, y, p: Population):
    """
    Stato y = [A_L, Cv]
      A_L : O2 alveolare totale (mL, STPD-equivalente come volume*frazione)
      Cv  : contenuto venoso misto di O2 (mL/dL) -> stato della riserva ematica
    """
    A_L, Cv = y
    A_L = max(A_L, 1e-6)
    Cv  = max(Cv, 1e-6)

    # --- pressione parziale alveolare di O2 dalla riserva polmonare ---
    frc_eff = p.frc_ml * (1.0 - p.f_trapped)    # FRC effettiva (closing capacity)
    fao2 = A_L / frc_eff                         # frazione O2 alveolare
    fao2 = min(max(fao2, 0.0), 1.0)
    pao2 = fao2 * (P_BARO - P_H2O)              # PAO2 (mmHg)

    # --- sangue capillare polmonare equilibrato con il gas alveolare ---
    scc = severinghaus_sao2(pao2)              # SatO2 fine-capillare
    ccc = o2_content(p.hb_gdl, scc, pao2)      # contenuto fine-capillare

    # --- saturazione/PO2 venosa dalla riserva (stato Cv) ---
    # inversione contenuto -> sat (trascurando il termine disciolto, piccolo)
    sv = np.clip((Cv - 0.0) / (HUFNER * p.hb_gdl), 1e-6, 1.0)
    pv = severinghaus_po2(sv)
    cv = o2_content(p.hb_gdl, sv, pv)

    # --- sangue arterioso = miscela shunt (venous admixture) ---
    s_shunt = shunt_at(t, p)                    # shunt tempo-dipendente
    ca = (1.0 - s_shunt) * ccc + s_shunt * cv  # contenuto arterioso
    # captazione polmonare di O2 (Fick): O2 che lascia l'alveolo verso il sangue
    j_up = p.co_dl_min * (ca - cv)             # mL/min   (CO in dL/min * mL/dL)
    j_up = max(j_up, 0.0)

    # --- flusso di massa apneico: gas richiamato dal faringe ---
    vco2_alv = p.r_apnea * p.vo2_ml_min        # CO2 aggiunta all'alveolo (piccola)
    j_net = max(j_up - vco2_alv, 0.0)          # gas netto richiamato (mL/min)
    if p.airway_pervio:
        o2_in_aw = j_net * p.f_aw_o2           # O2 entrante dalle vie aeree
    else:
        o2_in_aw = 0.0                         # vie chiuse: nessun rimpiazzo

    # --- bilanci ---
    dA_L = o2_in_aw - j_up                      # riserva polmonare
    # riserva ematica/tissutale: entra j_up (captazione), esce VO2 (metabolismo)
    # convertito in variazione di CONTENUTO venoso: dCv = (j_up - VO2)/Vol_sangue
    dCv = (j_up - p.vo2_ml_min) / p.vol_sangue_dl

    return [dA_L, dCv]

# =============================================================================
# SEZIONE 4 - RUNNER DI SIMULAZIONE + ESTRAZIONE OUTCOME
# =============================================================================

def simula(p: Population, t_max_s=1200.0, dt_s=1.0, rtol=1e-7, atol=1e-9,
           max_step=2.0):
    """Integra il modello e restituisce tempo (s), SpO2 (%), PAO2 (mmHg).
    Tolleranze allentabili per run Monte Carlo (rtol/atol/max_step)."""
    frc_eff = p.frc_ml * (1.0 - p.f_trapped)   # FRC effettiva (closing capacity)
    # condizioni iniziali
    A_L0 = p.fao2_iniz * frc_eff
    # riserva venosa iniziale: assumiamo SvO2 ~ 75% a regime pre-apnea
    pao2_0 = p.fao2_iniz * (P_BARO - P_H2O)
    sa0 = severinghaus_sao2(pao2_0)
    ca0 = o2_content(p.hb_gdl, sa0, pao2_0)
    cv0 = ca0 - p.vo2_ml_min / p.co_dl_min     # Fick a regime
    y0 = [A_L0, cv0]

    t_eval = np.arange(0.0, t_max_s + dt_s, dt_s)
    # le ODE sono in mL/min ma il tempo in secondi -> scala /60
    sol = solve_ivp(lambda t, y: [d / 60.0 for d in apnea_odes(t, y, p)],
                    [0.0, t_max_s], y0, t_eval=t_eval,
                    method="LSODA", rtol=rtol, atol=atol, max_step=max_step)

    A_L, Cv = sol.y
    fao2 = np.clip(A_L / frc_eff, 0, 1)
    pao2 = fao2 * (P_BARO - P_H2O)
    scc  = severinghaus_sao2(pao2)
    ccc  = o2_content(p.hb_gdl, scc, pao2)
    sv   = np.clip(Cv / (HUFNER * p.hb_gdl), 1e-6, 1.0)
    pv   = severinghaus_po2(sv)
    cv   = o2_content(p.hb_gdl, sv, pv)
    s_sh = shunt_at(sol.t, p)                   # shunt tempo-dipendente (array)
    ca   = (1 - s_sh) * ccc + s_sh * cv
    # SaO2 arteriosa dal contenuto (risolvi sat tale che content == ca)
    sa   = np.clip((ca - ALPHA_O2 * pao2) / (HUFNER * p.hb_gdl), 1e-6, 1.0)
    return sol.t, sa * 100.0, pao2

def tempo_a_soglia(t, spo2, soglia=90.0):
    """Tempo (s) al primo raggiungimento di SpO2 = soglia. None se mai."""
    sotto = np.where(spo2 <= soglia)[0]
    if len(sotto) == 0:
        return None
    i = sotto[0]
    if i == 0:
        return 0.0
    # interpolazione lineare
    t0, t1 = t[i-1], t[i]
    s0, s1 = spo2[i-1], spo2[i]
    return float(t0 + (soglia - s0) * (t1 - t0) / (s1 - s0))

# =============================================================================
# SEZIONE 5 - METRICHE DI AGREEMENT (pre-specificate)
# =============================================================================

def metriche_agreement(osservati, predetti):
    """Bias, RMSE, MAE, limiti di Bland-Altman."""
    o = np.array(osservati, float)
    pr = np.array(predetti, float)
    diff = pr - o
    bias = np.mean(diff)
    sd   = np.std(diff, ddof=1) if len(diff) > 1 else 0.0
    return {
        "n": len(o),
        "bias_s": bias,
        "rmse_s": float(np.sqrt(np.mean(diff**2))),
        "mae_s":  float(np.mean(np.abs(diff))),
        "loa_inf_s": bias - 1.96 * sd,
        "loa_sup_s": bias + 1.96 * sd,
    }

if __name__ == "__main__":
    # smoke test minimo
    p = Population(nome="adulto sano", eta_anni=35, peso_kg=70,
                   frc_ml=2300, vo2_ml_min=250, hb_gdl=15, co_dl_min=50,
                   shunt=0.05, fao2_iniz=0.87, airway_pervio=False, f_aw_o2=0.0)
    t, spo2, pao2 = simula(p, t_max_s=900)
    print(f"{p.nome}: t->SpO2 90% = {tempo_a_soglia(t, spo2):.0f} s ; "
          f"SpO2 finale {spo2[-1]:.1f}% ; PAO2 iniz {pao2[0]:.0f} mmHg")
