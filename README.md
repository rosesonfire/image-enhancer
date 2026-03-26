# image-enhancer
Naive approach to enhance image resolution

## Installation

### Prerequisites

- Python 3.13+

### Setup

1. Install dependencies:
```bash
pip install -e .
```

## Caution

Larger images will take longer time

## Usage

Before running, update the `FILE_PATH` and `SCALE` variables in `src/config.py` to point to your image file.

Run the prediction script:

```bash
./run.sh
```

#### Before enhancement

<img src="assets/demo_orig.jpg" alt="Before enhancement" width="2041" height="2041"/>

#### After enhancement with `scale=4`

<img src="assets/demo_enhanced.png" alt="After enhancement" width="2041" height="2041"/>
