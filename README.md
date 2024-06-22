# morphological-granulometric-analysis
Full code and resources of research project: granulometric analysis of maltodextrin particles observed by scanning electron microscopy.

The Curvature Analysis Method (CAM) is a morphological-based segmentation algorithm developed to detect circular shapes in 2D grayscale images for granulometry tasks (fig. below). Check out folder "curvature_analysis_method" for code.  
A sample of twenty real images of maltodextrin particules observed by scanning electron microscopy given as the object of study for this research project is provided to test the CAM. Check out folder "real_images_sample".  
A stochastic grains simulation model has also been created to generate realistic random images of grains for which the ground truth is known, aiming to validate the CAM. Check out folder "stochastic_grains_simulation" for code.  

The related study showed that the CAM gives much more accurate segmentation results, in terms of detection accuracy and Particule Size Distribution (PSD), than the traditional Circular Hough Transform (CHT) and the Stochastic Watershed (SW) give. These results are provided in the related papers published (see below).

Other ressources are available in folder "work_production": presentation slides and two posters recapping the main lines of the project.

Three minutes pitch for school is available at [Youtube - Mines Saint-Etienne](https://www.youtube.com/watch?v=pI0GmKkgZ7w)

Communication:
* Antoine Bottenmuller et al., "Granulometric Analysis of Maltodextrin Particles Observed by Scanning Electron Microscopy", 2023 IEEE 13th International Conference on Pattern Recognition Systems (ICPRS), Guayaquil, Ecuador, 2023, pp. 1-7, doi: 10.1109/ICPRS58416.2023.10179067
* Antoine Bottenmuller et al., "Une approche pour l’analyse granulométrique de particules condensées sur des images en niveaux de gris", 2023 29ème Colloque Francophone de Traitement du Signal et des Images (GRETSI), Grenoble, France, 2023, pp. 1153-1156, web: [HAL EMSE CNRS - 04195651](https://hal-emse.ccsd.cnrs.fr/emse-04195651/)

![alt text](https://raw.githubusercontent.com/antoine-bottenmuller/morphological-granulometric-analysis/main/CAM_example.png)
