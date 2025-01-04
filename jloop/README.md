# jloop
A small bash script for batch processing and task submission.  

The core idea is to keep VASP input files and submission scripts in one path. Modify these input files before submission, then copy input files and submission script to the task directions in batches and then submit tasks in batches. This method can also be applied to the batch submission of other program tasks.   

# Overview

- clog: View task submission logs.
- files: The direction where log files, input files, and scripts are stored.
- jgc: Grab information from the VASP output file.
- jhelp: Check help information for each script.
- jinsef: Check the maximum force and energy changes during VASP convergence.
- jloop: Perform various batch operations.
- jrunj: Submit the job and log it.
- pre: Copy the file to folder `files/` and add the corresponding call command
- put: Copy the files in folder `files/` to the current directory
- setup: Quick Setup Program.
- sw: View and modify scripts

# Installation
- Copy all the files to a folder.
- Modify pot_path and check_jobs value in setup script.
- Execute following command: `bash setup`
- Prepare a submit script and command, e.g. `pre -r xxxx/xxx.sh std54_56`

# Usage
Suppose you have prepared a series of vasp structure files xxx.vasp and want to batch submit jobs.
Start by using pre to modify INCAR and KPOINTS
```
pre -v INCAR
pre -v KPOINTS
```
Then you can run the following command to submit tasks in batches:
```
jloop dsuf . mtd put i,k apot run std54_56
```
vaspkit can also be used to generate KPOINTS and POTCAR:
```
jloop dsuf . mtd put i vkit p,m3 run std54_56
```
You can also submit only a few tasks at a time, such as the first twenty:
```
jloop dsuf . mtd put i,k vkit p
jloop h 20 run std54_56
```
You can also use nohup to submit tasks to the background:
```
nohup jloop runp 5 std54_56 > nohupoutput 2>&1 &
```
To submit jobs except VASP, prepare input script at first, then try:
```
jloop mtdz $input-file-name put $other-input-file run $job-submission-script
```

There are also other convenient features, such as querying the number and name of jobs that do not converge:
```
jloop ncon count name
```
Delete tasks submitted by current users:
```
jloop jid zdy
scancel $i
```
More features can be found in jhelp, and you can also add more custom functions to it.