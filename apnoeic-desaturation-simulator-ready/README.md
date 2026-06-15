# In-silico apnoeic oxygen desaturation simulator

This repository contains a deterministic physiological simulator of apnoeic oxygen desaturation, developed for educational and research use in paediatric anaesthesia modelling.

The model is a two-compartment mechanistic system describing oxygen mass balance during apnoea:

- an alveolar oxygen store, approximated by functional residual capacity (FRC);
- a lumped blood-tissue oxygen store;
- coupling through the haemoglobin oxygen dissociation curve, venous admixture, the Fick principle and apnoeic mass flow.

The repository includes:

- a standalone browser-based HTML simulator;
- an editable Jupyter notebook;
- Python validation scripts;
- benchmarking data and figures used in the accompanying manuscript.

## Important disclaimer

This simulator is for **educational and research use only**.

It does **not** predict individual patient risk, does **not** define safe apnoea time, and must **not** be used to guide airway management, oxygenation management, anaesthetic induction, or any clinical decision.

Outputs should be interpreted as:

> predicted time under model assumptions

and not as:

> safe apnoea time.

## Repository structure

```text
apnoeic-desaturation-simulator/
│
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
│
├── simulator/
│   └── apnea_desaturation_simulator.html
│
├── notebooks/
│   └── apnea_desaturation_editable_notebook.ipynb
│
├── src/
│   ├── apnea_desaturation_model.py
│   ├── validazione.py
│   └── montecarlo.py
│
├── figures/
│   ├── figure1_age_gradient_thrive.png
│   ├── figure2_validation_closing_capacity.png
│   ├── figure3_montecarlo.png
│   └── figureS1_dynamic_shunt.png
│
└── data/
    └── observed_benchmark_data.csv
```

## Quick start

### 1. Browser simulator

Open the standalone HTML file directly in a browser:

```text
simulator/apnea_desaturation_simulator.html
```

No installation is required.

### 2. Jupyter notebook

Install dependencies:

```bash
pip install -r requirements.txt
```

Then launch Jupyter:

```bash
jupyter notebook
```

Open:

```text
notebooks/apnea_desaturation_editable_notebook.ipynb
```

### 3. Python scripts

From the repository root:

```bash
python src/validazione.py
python src/montecarlo.py
```

The scripts regenerate the main validation and Monte Carlo figures.

## Model status

The baseline model is deterministic and a-priori-parameterised. It is not a statistical learner and it is not fitted to the validation cohorts.

Implemented modules include:

- baseline closed-airway apnoea;
- apnoeic oxygenation / THRIVE-like limiting case;
- optional age-dependent closing-capacity module;
- optional dynamic absorption-atelectasis shunt module;
- Monte Carlo propagation of interindividual physiological variability.

## Citation

If you use this software, please cite the archived release DOI and the accompanying manuscript.

A citation template is provided in `CITATION.cff`.

## License

The source code is released under the MIT License.

Manuscript text, figures and educational material may be distributed under a separate Creative Commons license if desired.
