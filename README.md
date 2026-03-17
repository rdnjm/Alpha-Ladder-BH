# Alpha Ladder -- Black Hole Phenomenology

A side quest from the [Alpha Ladder](https://github.com/rdnjm/Alpha-Ladder) framework. We wandered into the black hole sector to see if the omega = 0 Kaluza-Klein reduction had anything interesting to say about astrophysical observables. It did -- just not what we hoped.

## The Punchline

We explored six categories of black hole observables, computed everything honestly, and the universe said "no" in three different ways:

1. **Black holes are neutral.** Wald mechanism + Schwinger discharge keep q = Q/Q_ext below 10^-9. Every Gibbons-Maeda effect scales as q^2, pushing deviations below 10^-30 ppm. Not "hard to detect" -- literally unmeasurable by any instrument that could ever be built.
2. **Cassini already settled this.** The massless dilaton predicts a PPN gamma that disagrees with GR by ~0.2. Cassini measured it to 2.3 x 10^-5. That is a 20,000-sigma exclusion. The dilaton must be massive.
3. **Our own framework finishes the job.** Alpha Ladder flux stabilization gives the dilaton a Planck-scale mass (~6.3 x 10^29 eV). At that mass it decouples from everything, and all black hole solutions quietly revert to standard GR.

The real testable prediction was never about black holes. It is the sub-ppm prediction of G from fundamental constants: alpha^24 * mu * (mu - sqrt(phi) * (1 - alpha)) at -0.31 ppm with zero fitted parameters. That lives in the [main repo](https://github.com/rdnjm/Alpha-Ladder).

## Live Dashboard

Explore the full interactive dashboard at **[alpha-ladder-bh.streamlit.app](https://alpha-ladder-bh.streamlit.app/)**

Pick a black hole mass, dial the charge ratio, and watch every observable respond in real time. Spoiler: none of it matters at realistic charge levels.

## The Quest Log

| Page | Title | What happens |
|------|-------|-------------|
| 00 | Overview | The map and the spoiler |
| 01 | Gibbons-Maeda | We meet the boss: exact charged dilaton BH metric, horizons, thermodynamics |
| 02 | Quasinormal Modes | We listen to the ringdown. LIGO cannot hear the difference. |
| 03 | Shadows & EHT | The shadow shrinks 50% faster than RN. EHT still cannot tell. |
| 04 | ISCO & Accretion | Orbits shift, efficiency changes. At q ~ 0, nobody notices. |
| 05 | Observational Constraints | Cassini delivers the killing blow |
| 06 | Greybody & Hawking | The dilaton emission channel is kinematically blocked. Door shut. |
| 07 | The Verdict | Three stacked null results, one honest conclusion, and what IS testable |

## Project Structure

```
Alpha-Ladder-BH/
  app/
    Home.py                          # Streamlit entry point
    components/
      sidebar.py                     # Shared sidebar + global CSS
      charts.py                      # 10 Plotly chart builders
      formatting.py                  # Number formatting helpers
    pages/
      00_Overview.py ... 07_Verdict.py
  gibbons_maeda.py                   # GM metric, horizons, temperature, entropy
  quasinormal_modes.py               # QNM spectrum via WKB
  shadows.py                         # Photon sphere, shadow size, EHT constraints
  isco_accretion.py                  # ISCO, efficiency, luminosity
  observational_constraints.py       # Charge limits, PPN, effects scaling
  greybody_factors.py                # Hawking spectrum, greybody factors, dilaton channel
  requirements.txt
```

## Running

```bash
pip install -r requirements.txt
streamlit run app/Home.py
```

The dashboard launches in wide layout with an interactive sidebar for picking your black hole mass and dialing the charge ratio up and down. Spoiler: turning the charge knob past 10^-9 is pure fiction.

## Requirements

- Python 3.10+
- streamlit
- plotly

## What We Learned Along the Way

- **Dilaton coupling:** a = 1/sqrt(3), pinned by the omega = 0 reduction. Not negotiable.
- **Finite extremal temperature:** GM black holes keep radiating at extremality. RN black holes freeze. Nature chose neither for real black holes (q ~ 0), so this is trivia.
- **Shadow shrinkage:** ~50% faster than Reissner-Nordstrom. A beautiful theoretical result that astrophysics renders moot.
- **Everything scales as q^2:** Which means everything scales as (basically zero)^2.
- **Dilaton channel blocked:** Planck-mass dilaton cannot fit through the Hawking emission window. Kinematically forbidden.

The side quest was worth it. Honest null results are still results, and now we know the black hole sector is clean -- no contradictions, no surprises, no observable consequences. The interesting physics is in the G prediction.

## License

Apache 2.0 -- see [LICENSE](LICENSE).
