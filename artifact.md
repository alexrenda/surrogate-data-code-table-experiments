# Artifact Documentation for Turaco: Complexity-Guided Data Sampling for Training Neural Surrogates of Programs

This repository contains the implementation of the Turaco programming language and its analysis, and the experiments in the paper "Turaco: Complexity-Guided Data Sampling for Training Neural Surrogates of Programs".

## Installation

Requires Python >= 3.9.

First create a virtual environment and install the requirements:
```
python -m venv env
. env/bin/activate
pip install -r requirements.txt
```
Then install Turaco:
```
pip install -e analysis
```

### Renderer

The renderer example also requires the following dependencies:

* g++ (GCC) 11.1.0
* Panda3D 1.10.9
* OpenGL version string: 4.6.0 NVIDIA 465.31

The environment variables `P3D_INCLUDE_PATH` and `P3D_LIB_PATH` must be set to point to the Panda3D library.

## Artifact Contents

* The Turaco language and analysis (`turaco/analysis/turaco`)
* Code for experiments in Sections 2 and 5.1 (`turaco/example-experiments`) and Section 5.2 (`turaco/renderer-experiments`)

### Computing Complexities

The `example-experiments` directory has several examples, including the examples from Sections 2 and 5.1.

To run the standard interpretation of Turaco on a program, run:
```
python turaco/turaco.py --program {{PROGRAM_FILE}} interpret --input INPUT_NAME:INPUT_VALUE INPUT_NAME:INPUT_VALUE ...
```

For example, to execute the example in Section 2 of the paper, run:
```
python turaco/turaco.py --program luminance.t interpret --input sunPosition:0.5 emission:0.2
```

To run the complexity interpretation of Turaco on a program, run:
```
python turaco/turaco.py --program {{PROGRAM_FILE}} complexity --input BETA
```

For example, to calculate the complexity of the daytime example in Section 2 of the paper, run
```
python turaco/turaco.py --program luminance.t complexity --input 1
```

### Computing Sampling Distributions

The file `plot.py` in `example-experiments` contains utilities for computing the complexity-guided sampling distribution for a given program.

For example:
```
python3 plot.py --program luminance.t --print-dist
```
will print the computed complexity of each path (as a dictionary labeled `Complexity`), then read the per-path frequency distribution from `luminance.t.yaml` and print the computed per-path sampling distribution (as a dictionary labeled `optimal`).

### Evaluating Accuracy Improvements

To support evaluation, we provide the full set of trained neural networks, instructions to perform a representative evaluation in one setting, and instructions to re-train all neural networks from scratch to perform the full evaluation.

#### Evaluating Using Pre-Trained Networks

This section shows how to use pre-trained networks to reproduce the results in the paper.

##### Section 5.1: Evaluation Across Programs

The working directory for this experiment is `turaco/example-experiments`.

To evaluate using pre-trained neural networks, run `python3 table.py`. This will print out Table 1:
```
\multirow{2}{1.5cm}{\bf Benchmark} & \multirow{2}{1.5cm}{\centering\textbf{Path}} & \multirow{2}{2cm}{\centering\textbf{Complexity}} & \multirow{2}{2cm}{\centering\bf Frequency \\ Distribution} & \multirow{2}{2cm}{\centering\bf Uniform \\ Distribution} & \multirow{2}{2cm}{\centering\bf Complexity \\ Distribution} \\&&&&\\ \midrule
\toprule
Benchmark & Lines of Code & Paths & Frequency (Predicted) & Frequency (Measured) & Uniform (Predicted) & Uniform (Measured) \\ \midrule
Luminance & $14$ & $3$ & $2.58\%$ & $15.01\%$ & $6.97\%$ & $15.17\%$ \\
Huber & $13$ & $3$ & $0.49\%$ & $8.15\%$ & $1.93\%$ & $9.54\%$ \\
BlackScholes & $15$ & $2$ & $4.43\%$ & $3.61\%$ & $1.30\%$ & $4.00\%$ \\
Camera & $69$ & $3$ & $2.83\%$ & $0.56\%$ & $0.22\%$ & $1.36\%$ \\
EQuake & $34$ & $2$ & $7.45\%$ & $2.25\%$ & $7.45\%$ & $2.25\%$ \\
Jmeint & $176$ & $18$ & $2.34\%$ & $0.01\%$ & $8.44\%$ & $1.02\%$ \\
\midrule
Geomean & & & $3.33\%$ & $4.81\%$ & $4.33\%$ & $5.43\%$ \\
\bottomrule
```

##### Section 5.2: Renderer Demonstration

The working directory for this experiment is `turaco/renderer-experiments`.

To build, run `./build.sh`. To run, run `./3d-game-shaders-for-beginners`. To run with a surrogate, run:
```
./3d-game-shaders-for-beginners --surrogate surrogates/base_day_base_day_optimal_61_2.data
```

#### Representative Small-Scale Evaluation

A tractable small-scale evaluation is evaluating surrogates at a single dataset budget in the Luminance example (Section 2). This will reproduce results for a single x value in Figure 2 (b).

The working directory for this experiment is `turaco/example-experiments`. Note that running this evaluation will overwrite the pre-trained surrogate; if this is a concern, back up the directory `data/luminance.t`.

First, calculate the complexity and sampling ratios for each distribution by running `python3 plot.py --program luminance.t --print-dist`, which will print out:
```
Complexity : {'luminance.t:ll': 0.010000000000000002, 'luminance.t:rl': 1.2100000000000002, 'luminance.t:rr': 9.0}
optimal: {'luminance.t:ll': 0.3694326245766699, 'luminance.t:rl': 0.13982267657067707, 'luminance.t:rr': 0.49074469885265304} (n=1000: {'luminance.t:ll': 369.4326245766699, 'luminance.t:rl': 139.82267657067706, 'luminance.t:rr': 490.744698852653})
data: {'luminance.t:ll': 0.5, 'luminance.t:rl': 0.1, 'luminance.t:rr': 0.4} (n=1000: {'luminance.t:ll': 500.0, 'luminance.t:rl': 100.0, 'luminance.t:rr': 400.0})
```
The `optimal` dictionary shows the fraction of data to be sampled from each path for the complexity-based distribution, and the `data` dictionary shows the fraction of data for the frequency distribution.

We will evaluate at n=50. This will involve training three surrogates (one per path) for each distribution.

First we will train for the complexity distribution. This involves training a surrogate on the `ll` path with `0.369*50` samples, the `rl` path with `0.140*50` samples, and the `rr` path with `0.491*50` samples:
```
python3 driver.py --program luminance.t train --n 18 --trial 0 --lr 5e-5 --steps 10000 --depth 1 --path ll
python3 driver.py --program luminance.t train --n 7 --trial 0 --lr 5e-5 --steps 10000 --depth 1 --path rl
python3 driver.py --program luminance.t train --n 25 --trial 0 --lr 5e-5 --steps 10000 --depth 1 --path rr
```
which (if ran in sequnce) will result in:
```
ll   : Test Prob 0.500 | Train Prob 1.000 | Loss 4.430e-03
rl   : Test Prob 0.100 | Train Prob 1.000 | Loss 1.022e-02
rr   : Test Prob 0.400 | Train Prob 1.000 | Loss 7.139e-03
```
Note that depending on your platform, you may have slightly different loss values.

Next, train on the frequency distribution:
```
python3 driver.py --program luminance.t train --n 25 --trial 0 --lr 5e-5 --steps 10000 --depth 1 --path ll
python3 driver.py --program luminance.t train --n 5 --trial 0 --lr 5e-5 --steps 10000 --depth 1 --path rl
python3 driver.py --program luminance.t train --n 20 --trial 0 --lr 5e-5 --steps 10000 --depth 1 --path rr
```
which results in:
```
ll   : Test Prob 0.500 | Train Prob 1.000 | Loss 3.520e-03
rl   : Test Prob 0.100 | Train Prob 1.000 | Loss 3.874e-02
rr   : Test Prob 0.400 | Train Prob 1.000 | Loss 8.959e-03
```

Together, the error (according to the frequency distribution) of the complexity-guided surrogate is `0.5*4.430e-03 + 0.1*1.022e-02 + 0.4*7.139e-03 = 0.0060926`. The error of the frequency surrogate is `0.5*3.520e-03 + 0.1*3.874e-02+0.4*8.959e-03 = 0.0092176`. The complexity-guided surrogate decreases the error relative to the frequency surrogate by 34%, consistent with the trend of the results in Figure 2(b).

#### Full Re-Training

##### Section 5.1: Evaluation

The working directory for this experiment is `turaco/example-experiments`.

First, move/remove the pre-trained neural networks: `mv data data.pretrained; mv datasets datasets.pretrained`.

For Table 3, to train the neural networks from scratch, run `python3 table.py` (this may take several hours to days to run, depending on the CPU). This will print out a table similar to Table 3. Though PyTorch nondeterminism might make the numbers slightly different, the broad trend of complexity-based sampling decreasing error relative to frequency-based sampling should hold.

### Evaluating the Renderer

To run the examples in Section 5:
```
cd renderer-experiments
pushd renderer
./build.sh
for s in base_day base_night top_day top_night; do ./3d-game-shaders-for-beginners --log --frames 100 --scene $s; done
popd
```
Each scene will print out an identifier like `166XXXXXXXXXXXX`, which we will call `IDENTIFIER`. Copy `textures_IDENTIFIER` to `renderer-experiments/data/textures_IDENTIFIER`. Then:
```
python main.py --identifier IDENTIFIER dataset
```
To construct a stratified surrogate at a given number of data points:
```
python main.py --identifier IDENTIFIER --from 1 --to 6 --steps 20 generate --n NUMBER_OF_SAMPLES --trial 0 --type (optimal/test/uniform)
```
The above command writes a file `data/surrogates/(optimal/test/uniform)_NUMBER_OF_SAMPLES_0.data`, which we will call `SURROGATE_FILE`.
To run this in the renderer:
```
cd renderer-experiments
pushd renderer
./3d-game-shaders-for-beginners --surrogate SURROGATE_FILE
```
