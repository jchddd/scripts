# scripts
Custom scripts mainly for VASP calculation and catalysis related content.
## jworkflow
Custom Python scripts based mainly on **pymatgen** and are used to analyze VASP calculation results and catalysis related content
### Requirement
- pymatgen
- ase
- sklearn
- matplotlib
- numpy
- pandas
- openpyxl
### Overview
- slab: Cleave surface, find adsorption sites, and add adsorbates in batches. Modify the slab structure by layer.
- dada_process.py: Extract VASP calculation results like slab relaxation, adsorbate adsorption energy and free energy correction. Process  POSCAR files for frequency and differential charge density calculations.
- reaction.py: Calculate adsorption energy, find stablest adsorption structure from VASP calculations, find potential determinted step and calculate energy profile.
- plot.py: Plot slab structures and energy profile.
- electronic.py: Gather DOS and COHP information from vasprun.xml and cohpcar.lobster file respectively.
- dataset.py: Store molecular structures and energies, and reaction formulas.
