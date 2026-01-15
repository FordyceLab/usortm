# Cost Data

This directory contains pre-computed cost data generated from the Python cost functions in `src/usortm/costs/cost_functions.py`.

## Files

- `cost_curve_300bp.json` - Cost curve for 300bp sequences
- `cost_curve_500bp.json` - Cost curve for 500bp sequences
- `cost_curve_750bp.json` - Cost curve for 750bp sequences
- `cost_curve_1000bp.json` - Cost curve for 1000bp sequences
- `default_costs.json` - Default configuration (500 variants, 300bp)

## Regenerating Data

If cost functions are updated in the Python source, regenerate this data:

```bash
cd docs
python3 generate_cost_data.py
```

This ensures the web documentation uses the exact same cost calculations as the Python package.
