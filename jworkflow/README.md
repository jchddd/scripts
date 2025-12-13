# jworkflow
Custom Python scripts based mainly on **pymatgen** and are used to analyze VASP calculation results and catalysis related content
## Installation
Copy the Folder Jworkflow into the Lib folder under your Python directory, or run the code at the same level as Folder Jworkflow.
## Requirement
- pymatgen
- ase
- sklearn
- matplotlib
- numpy
- pandas
- openpyxl
## Overview
- [slab.py](https://github.com/jchddd/scripts/blob/main/jworkflow/Jworkflow/slab.py): Cleave surface, find adsorption sites, and add adsorbates in batches. Modify the slab structure by layer.
- [dada_process.py](https://github.com/jchddd/scripts/blob/main/jworkflow/Jworkflow/data_process.py): Extract VASP calculation results like slab relaxation, adsorbate adsorption energy and free energy correction. Process  POSCAR files for frequency and differential charge density calculations.
- [reaction.py](https://github.com/jchddd/scripts/blob/main/jworkflow/Jworkflow/reaction.py): Calculate adsorption energy, find stablest adsorption structure from VASP calculations, find potential determinted step and calculate energy profile.
- [plot.py](https://github.com/jchddd/scripts/blob/main/jworkflow/Jworkflow/plot.py): Plot slab structures and energy profile.
- [electronic.py](https://github.com/jchddd/scripts/blob/main/jworkflow/Jworkflow/electronic.py): Gather DOS and COHP information from vasprun.xml and cohpcar.lobster file respectively.
- [dataset.py](https://github.com/jchddd/scripts/blob/main/jworkflow/Jworkflow/dataset.py): Store molecular structures and energies, and reaction formulas.
- [pourbiax.py](https://github.com/jchddd/scripts/blob/main/jworkflow/Jworkflow/pourbiax.py): Plot Pourbiax diagram and store example data.
## Tutorial
See Tutorial Jupyter Notebooks

