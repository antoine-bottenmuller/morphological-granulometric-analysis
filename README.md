# morphological-granulometric-analysis
Full code and ressources of the research project: granulometric analysis of maltodextrin particles observed by scanning electron microscopy

The Curvature Analysis Method (CAM) is a morphological-based segmentation method developed to detect circular shapes in grayscale images for granulometry tasks. Check folder "curvature_anaylis_method" for code.
A sample of twenty real images of maltodextrin particules observed by scanning electron microscopy given as the object of study for this research project is provided to test the CAM. Check folder "real_images_sample".
A stochastic grains simulation model has also been created to generate realistic random images of grains for which the ground truth is known, aiming to validate the CAM. Check folder "random_grains_generator" for code.

The related study showed that the CAM gives much more accurate segmentation results, in terms of detection accuracy and Particule Size Distribution (PSD), than the traditional Circular Hough Transform (CHT) and the Stochastic Watershed (SW) give. These results are provided in the related papers published (see below).

Other ressources are available in folder "other_ressources": presentation slides and two posters recapping the main lines of the project.

Three minutes pitch for school available at [Youtube - Mines Saint-Etienne](https://www.youtube.com/watch?v=pI0GmKkgZ7w)

Papers published:
* Antoine Bottenmuller et al., "Granulometric Analysis of Maltodextrin Particles Observed by Scanning Electron Microscopy," 2023 IEEE 13th International Conference on Pattern Recognition Systems (ICPRS), Guayaquil, Ecuador, 2023, pp. 1-7, doi: 10.1109/ICPRS58416.2023.10179067
* Antoine Bottenmuller et al., "Une approche pour l’analyse granulométrique de particules condensées sur des images en niveaux de gris", 2023 29ème Colloque Francophone de Traitement du Signal et des Images (GRETSI), Grenoble, France, 2023, pp. 1153-1156, web: [HAL EMSE CNRS - 04195651](https://hal-emse.ccsd.cnrs.fr/emse-04195651/)
