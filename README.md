# orthoPE

A Python library for computing Orthographic Prediction Error (oPE) estimates based on predictive coding models of visual word recognition.

This library implements multiple oPE models, including several beyond those described in the original publication, to analyze how visual and orthographic features predict behavioral and neural responses during word recognition.

## Features

- **Multiple oPE Estimators**: 7 different estimation methods including L1/L2 norms, Mahalanobis distance, and Kalman-weighted prediction error
- **Flexible Text Rendering**: Pixel-based rendering with TrueType fonts or abstract letter-based encoding
- **Noise Simulation**: Configurable Gaussian noise to model perceptual uncertainty
- **Multi-Language Support**: German (fully tested), with experimental support for English, French, and Dutch
- **Analysis Tools**: Built-in correlation analysis and visualization functions for behavioral (RT) and neural (EEG) data

## Installation

```bash
pip install numpy pandas scipy matplotlib seaborn pillow tqdm
```

Clone the repository:
```bash
git clone https://github.com/yourusername/orthoPE.git
cd orthoPE
```

## Fonts

The library supports rendering with TrueType fonts. Sample fonts used in testing include Courier, Cambria, and Verdana. Due to licensing restrictions, fonts are not included in this repository.

To use font-based rendering:
1. Download TrueType fonts (.ttf files) from legitimate sources
2. Place them in a `./fonts/` directory in the project root
3. Additional fonts can be added but have not been systematically tested

**Note**: The package can run without proprietary fonts using the letter-based encoding mode (`font='word'`), which uses abstract orthographic representations instead of pixel rendering.

## Data

Corpus and experimental data are not included in this repository. The data used in this repository is posted in https://osf.io/d8yjc/. These data is part of the research paper https://doi.org/10.1016/j.neuroimage.2020.116727, which should be cited if using the data.

Required data structure:
- **Corpus files**: CSV with word frequencies (e.g., `ger5_fin.csv`)
- **Behavioral data**: CSV with reaction time data from lexical decision tasks
- **EEG data** (optional): CSV with neural responses at different time windows

Place data files in `./data_repository/` directory.

## Quick Start

### Compute oPE estimates

```python
import orthope

# Pixel-based rendering with specific font
orthope.run_all_oPEs('german', 'courier')

# Letter-based encoding (no font required)
orthope.run_all_oPEs('german', 'word')
```

### Run full analysis pipeline

```python
import pipelines

# Compute models and generate visualizations
pipelines.compute_all_models('german', fonts=['courier'])
pipelines.compute_all_models('german', 'word')

# Integrate multiple models into single CSV
modeldict = {
    'pred_err_l2': 'PE2',
    'mahalanobis': 'Mahalanobis',
    'kalmanw_pred_err': 'Kalman'
}
pipelines.integrate_models_in_csv(modeldict, 'all_models')
```

### Load and analyze results

```python
import orthope

# Initialize estimator
gg = orthope.OrthopeEstimator('german', 'courier', noise=0.1)

# Load computed oPE estimates
opes_df = gg.load_opes()

# Load behavioral data
rt_df = gg.load_data_rt()

# Load EEG data (German only)
eeg_df = gg.load_data_eeg()

# Compute correlations
correlations = orthope.compute_oPE_RT_correlations('german', 'courier')

# Generate visualizations
orthope.plot_oPE_RT_scatterplots('german', 'courier')
orthope.plot_oPE_RT_rhos('german', 'courier')
```

## oPE Estimator Types

The library implements 7 oPE estimation methods:

1. `n_pixels_l1` - L1 norm of pixel differences
2. `n_pixels_l2` - L2 norm of pixel differences
3. `pred_err_l1` - L1 prediction error from corpus statistics
4. `pred_err_l2` - L2 prediction error from corpus statistics
5. `pw_pred_err` - Pixel-wise prediction error
6. `mahalanobis` - Mahalanobis distance using corpus covariance
7. `kalmanw_pred_err` - Kalman-weighted prediction error

## Testing

Run basic regression tests to ensure nothing is broken:

```bash
python test_basic.py
```

This runs 19 fast tests (~0.5 seconds) covering core functionality.

## Language Support

- **German**: Fully tested and validated
- **English, French, Dutch**: Experimental support (not systematically tested)

Use non-German languages with caution and validate results independently.

## Output

The library generates:
- **Models**: CSV files with oPE estimates (`./models/`)
- **Results**: Correlation plots and scatterplots (`./results/`)
- **Integrated data**: Combined model outputs (`all_models.csv`)

## Advanced Usage

### Custom noise levels

```python
# Initialize with specific noise level
gg = orthope.OrthopeEstimator('german', 'courier', noise=1.5)

# Available noise levels: [0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0]
```

### Letter-based encoding

```python
# Use abstract orthographic representation instead of pixels
wg = orthope.LetterOrthopeEstimator('german', noise=0.1)
result = wg.__render_text__('hello')
```

### Compute single oPE estimate

```python
gg = orthope.OrthopeEstimator('german', 'courier', noise=0.1)
gg.corpus_stats = gg.__estimate_corpus_stats__()
ope_value = gg.__estimate_ope__('example', estimator='mahalanobis')
```
