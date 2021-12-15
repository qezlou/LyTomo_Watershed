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


### Generated Data

- The generated data are available here [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5770883.svg)](https://doi.org/10.5281/zenodo.5770883). 
- Refer to the `CookBook.ipynb` and other notebooks to see which data you need at each step. 
- A clear descitption of each file is provided on the Zenodo website. 

- The data should be in a directory named `LyTo_data` outside this repository. So, it should look like this:

```
./LyTomo-Wtershed/
     codes/
     *.ipynb
./LyTomo_data/
     descendants/
     mock_maps_z2.4/
     ...
```

- You can use `get_data.py` script to download the files from the shell.

     1. Get your access token from [here](https://zenodo.org/account/settings/applications/tokens/new/)
     
     2. To downlaod all the compressed files, run this on your shell:
     ```
     python get_data.py -t "Your ACCESS TOKEN"
     ```
     If you want to download a particular compressed file, pass the file name like this :

    ```
     python get_data.py -t "Your ACCESS TOKEN" -f 'descendats.zip'
     ```
     3. Don't forget to decompress the downloaded files.

If you have any questions please send me an email : mahdi.qezlou@email.ucr.edu or raise an issue here!

Note : The observed LATIS data [Newman et al. 2020](https://arxiv.org/abs/2002.10676) is not publicly released yet. 
       Please stay tuned for a near future paper!


- 
