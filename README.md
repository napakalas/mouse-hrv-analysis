# mouse-hrv-analysis

This code is a used to analyse mouse hrs recorded consecutively using AdInstrument telemetry device.

- install requirement.txt
- make sure jupyter is available
- run jupyter
- open `hrv_analysis.ipynb`
- put required mat files in `data` folder or modiy variable `data_path` with mat files location

Note:

- `signal` and `peaks` as the result of `get_data` method is not yet filter

To do:

- filter `peaks` and `signal`
- call pyhrv to do frequency and linear domains analysis
