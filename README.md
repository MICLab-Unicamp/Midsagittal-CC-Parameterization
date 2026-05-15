# Midsagittal Corpus Callosum Parameterization

This repository contains an implementation of the proposed corpus callosum (CC) parameterization method, designed for the analysis of the midsagittal CC from diffusion MRI data. The method resamples the structure using a standardized grid of sampling points, enabling direct point-by-point comparisons across individuals.

The approach is computationally efficient, avoids explicit nonlinear inter-subject registration and spatial smoothing, and provides a spatially detailed characterization of diffusion properties across most of the midsagittal CC.

## Paper

The manuscript describing this method is currently under submission.

## Input requirements

For each subject, the method requires the following inputs:

- A binary midsagittal CC mask
- Diffusion-derived maps (for example, FA, MD, RD, and AD maps)

### Recommended preprocessing

The method was developed and evaluated using diffusion MRI data preprocessed with a consistent pipeline. We recommend the following preprocessing steps before running the parameterization:

1. Denoising
2. Eddy-current and motion correction
3. Rigid alignment to the MNI152 template (translation and rotation only), to reduce variability in midsagittal slice selection caused by differences in head positioning
4. Resampling to 1.25 mm isotropic resolution
5. Brain extraction
6. Diffusion tensor reconstruction and diffusion maps generation
7. Midsagittal CC segmentation (or CC segmentation + midsagittal slice identification)

In our experiments, the midsagittal CC segmentation was obtained by combining automated [TractSeg](https://github.com/mic-dkfz/tractseg) bundle segmentation with targeted post-processing to isolate only the midsagittal section of the CC. The midsagittal slice was identified from the FA maps based on the diffusion characteristics of the interhemispheric fissure [[1]](https://doi.org/10.1117/12.911619).

#### Notes

The robustness of the method to alternative preprocessing strategies remains to be investigated. In particular, differences in voxel resolution may influence the choice of the parameterization configuration. All experiments in the associated manuscript were performed using diffusion images resampled to 1.25 mm isotropic resolution.

## Online demo

An online Streamlit demo is available, allowing you to test the parameterization method using your own uploaded images.

To access the demo, visit:

https://cc-parameterization.streamlit.app/

## Offline local tool

If you want to process large datasets or prefer not to upload your files, we recommend using the local version of the application. Installation and usage instructions are provided below.

### Requirements

The application was tested on Ubuntu 20.04 and Windows 11 using Python 3.10 in Miniconda environments. The required dependencies are listed in `requirements_local.txt`.

We recommend using a dedicated Conda environment to avoid interfering with existing Python installations. To install Miniconda on Windows or Linux, follow the instructions at:

https://docs.conda.io/en/latest/miniconda.html

On Windows, the following commands should be executed in the Anaconda Prompt.

Create a new environment with:

    conda create -n cc-param python=3.10

### Installation

Activate the environment:

    conda activate cc-param

Clone the repository:

    conda install git
    git clone https://github.com/MICLab-Unicamp/Midsagittal-CC-Parameterization

Go to the repository folder:

    cd Midsagittal-CC-Parameterization

Install the required packages:

    conda config --add channels conda-forge
    conda install --file requirements_local.txt

It may take a few minutes for Conda to solve the environment, especially on Windows.

### Running the application

Inside the repository folder, run:

    streamlit run local_app.py

The Streamlit application should open automatically in your browser. From there, just follow the instructions to use the method.

## Using the Streamlit interface

A few notes about the Streamlit application:

- The local application may exhibit slower performance on Windows than on Linux.
- To reset the application state and start a clean session, simply refresh the browser page.
- During processing, Streamlit shows an activity indicator in the top-right corner of the page. This interface also allows you to stop the current execution if necessary.
