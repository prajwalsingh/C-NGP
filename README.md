<h2 style="text-align:center;"> Incremental Multi-Scene Modelling via Continual Neural Graphics Primitives </h2> 
<p>[Paper] &nbsp;&nbsp; <a href="https://anonymous.4open.science/w/C-NGP/">[Website]</a></p>

<br/>

<image src="results/c3ngp_teaser.png" width="100%" height="400px"/>

### Conditional modification to NGP

<image src="results/cclngp_architecture.png" width="512px">

### Results

||
|-----|
|Nerf Real Synthetic 360|
|<image src="results/nerf_synth.png" width="100%"/>|
|Nerf Real Forward Facing|
|<image src="results/nerf_real.png" width="100%"/>|
|Tanks and Temple|
|<image src="results/qual_tanksandtemple.png" width="100%"/>|
<!-- |Qualitative Comparision|
|<image src="results/qual_comparision.png" width="100%"/>|
|<image src="results/extra_comparision.png" width="100%"/>| -->

### Mix Scene Shared NGP Space Visualization
||
|-----|
|Image|
|<image src="results/mixscene.png" width="512px" height="512px"/>|
|GIF|
|<image src="./results/mixscene.gif" width="512px" height="512px"/>|

### Style Editing

<image src="results/multistylengp.png" width="668px" height="400px">

### Python libraries [[reference](https://github.com/kwea123/ngp_pl/tree/master#software)]
  * Create a conda environment from shared environment.yml
  * Install `torch-scatter` following their [instruction](https://github.com/rusty1s/pytorch_scatter#installation)
  * Install `tinycudann` following their [instruction](https://github.com/NVlabs/tiny-cuda-nn#pytorch-extension) (pytorch extension)
  * Install `apex` following their [instruction](https://github.com/NVIDIA/apex#linux)
  * Install core requirements by `pip install -r requirements.txt`

* Cuda extension: Upgrade `pip` to >= 22.1 and run `pip install models/csrc/` (please run this each time you `pull` the code)

### Note

* We will release the checkpoints later. Due to the large size, we have to upload it on the drive, and it is not possible to share it without breaking anonymity.

### References
<pre>
  [1] Müller, T., Evans, A., Schied, C., & Keller, A. (2022). Instant neural graphics primitives 
  with a multiresolution hash encoding. ACM transactions on graphics (TOG), 41(4), 1-15.
  [2] InstantNGP Pytorch Implementation: https://github.com/kwea123/ngp_pl]
</pre>
