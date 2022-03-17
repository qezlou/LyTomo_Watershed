# LyTomo-Watershed
Characterizing proto-clusters/groups in Lyman-alpha Tomography

The repository for reproducing the results in [Qezlou et al. 2021](https://arxiv.org/abs/2112.03930)

### Cookbook :

There is a step-by-step cookbook in [`CookBook.ipynb`](https://github.com/mahdiqezlou/LyTomo-Watershed/blob/main/CookBook.ipynb)

### Requirements :

Each part of the production has a different package requirements. 
Please, review the imported packages in each section. A complete list is :


- numpy
- scipy
- scikit-image
- fake_spectra
- mpi4py

### Installing the package:

via pip:

` pip install lytomo_watershed`

or cloning this repo and :

` python3 setup.py install `

### Generated Data

- The generated data are available here [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5770882.svg)](https://doi.org/10.5281/zenodo.5770882). 
- Refer to the `notebooks/CookBook.ipynb` and other notebooks to see which data you need at each step. 
- A clear descitption of each file is provided on the Zenodo website. 

- You can use `get_data.py` script to download the files in your notebook:

     1. Get your Zenodo access token from [here](https://zenodo.org/account/settings/applications/tokens/new/)
     
     2. To downlaod any of the compressed files, run :
     ```
     import lytomo_watershed
     from lytomo_watershed import get_data
     get_data.download(token="YOUR_ACESS_TOKEN", data_dir='./', file='descendant.zip')
     # Set file=None, to downlaod everything
    ```
     3. Don't forget to decompress the downloaded files.

If you have any questions please send me an email : mahdi.qezlou@email.ucr.edu or raise an issue here!

Note : The observed LATIS data [Newman et al. 2020](https://arxiv.org/abs/2002.10676) is not publicly released yet. 
       Please stay tuned for a near future paper!


- 
