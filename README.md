# Collaborative Deep Reinforcement Learning (CDRL) for LEO Satellite Resource Management

This repository contains the implementation of the **proposed CDRL framework** for joint frequency reuse factor and transmit power control to maximize energy efficiency (EE) in LEO satellite communication networks.

The framework realizes inter-satellite knowledge sharing via **simultaneous neural network weight transfer and experience replay buffer transfer**, enabling energy-efficient resource management with improved convergence speed and learning stability.

## Paper

> *Jointly Optimizing Frequency Reuse and Transmit Power for LEO Satellite Networks via Collaborative DRL with Information Sharing.*
> Submitted to **IEEE Internet of Things Journal** (Manuscript ID: IoT-61639-2026).

## Repository Structure

```
.
├── MAIN_Proposed.py      # Main training script for the proposed CDRL method
├── Environment.py        # LEO satellite communication environment (channel, fading, SINR, EE)
├── parameters.py         # System / simulation hyperparameters
├── Unit.py               # Unit conversion utilities (dB <-> watt <-> dBm)
├── UserLocation.xlsx     # Ground user location data (per ground network)
├── UserVector.xlsx       # User mobility vector data
├── requirements.txt      # Python package dependencies
└── README.md
```

## Installation

```bash
# (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

## How to Run

```bash
python MAIN_Proposed.py
```

The script runs the proposed CDRL training over the configured number of episodes and satellites. Simulation results (per-network and per-satellite scores) are saved as `Proposed_test{timestamp}.xlsx` in the current directory.

## Key Configuration

Edit `parameters.py` to adjust system parameters:

| Parameter | Description | Default |
|---|---|---|
| `number_of_satellite` | Number of LEO satellites in the orbit | `49` |
| `number_of_user` | Number of ground users per network | `64` |
| `number_of_beam` | Number of beams per satellite | `8` |
| `number_of_iteration` | Episode length (timesteps) | `1000` |
| `kind_of_frequency_reuse` | Frequency reuse factor candidates | `[1, 2, 4, 8]` |
| `kind_of_power` | Discrete transmit power levels | `[0, P/4, P/2, 3P/4, P]` |
| `P_ISL_TX` | ISL transmit power [W] | `10.0` |
| `R_ISL` | ISL link capacity [bps] | `10e6` |
| `RHO_D` | CSI temporal correlation coefficient | `0` (worst-case) |

Hyperparameters of the DQN agent (learning rate, epsilon decay, batch size, etc.) are defined inside the `DQN` class in `MAIN_Proposed.py`.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{cho2026cdrl,
  title   = {Jointly Optimizing Frequency Reuse and Transmit Power for LEO
             Satellite Networks via Collaborative DRL with Information Sharing},
  author  = {Cho, Hyebin and Lee, In-Ho and Lee, Howon},
  journal = {IEEE Internet of Things Journal},
  year    = {2026},
  note    = {Under review}
}
```

## Contact

For questions about the code or the paper, please open an issue on this repository.

