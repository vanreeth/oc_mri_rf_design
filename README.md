# oc_mri_rf_design
Python-based generic framework for the optimal design of radiofrequency pulses in MRI

# Usage instructions
## Configure virtual environment
1. Get configuration file : grape_dependencies.yml
2. Create venv: 
    conda env create -f .\grape_dependencies.yml
3. Activate venv:
    conda activate grape-env

## Install wheel
1. Get in the projet local root folder 
2. Run "python -m build" (considering build package installed). 
3. Activate virtual environment: conda activate grape-env
4. Install wheel: "pip install --force-reinstall ./dist/grape-0.1.0-py3-none-any.whl"
