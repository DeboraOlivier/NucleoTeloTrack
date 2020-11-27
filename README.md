What is NucleoTeloTrack?
=================

NucleoTeloTrack is a combination of **image analysis and data analysis** tools to go from **live imaging data to figures**.
It consists of:
1) a **CellProfiler image analysis pipeline** that segments nuclei & telomeres, tracks nuclei, classifies them according to cell cycle stage, and perform various measurements (intensity, shape, counts)
2) a series of **Jupyter Notebooks** to extract various nuclei tracks, align them according to a reference time, and generate graphs for multiple parameters including statistical anaysis.

Contributors
=================
The CellProfiler pipeline was created by Debora Keller [OrcidID](https://orcid.org/0000-0002-5284-3195) in [Laure Crabbe's lab](https://sites.google.com/yahoo.fr/crabbelab/).

The jupyter notebooks were developped by [Minh-Son Phan](https://msphan.github.io/) and Anatole Chessel (Assistant Professor, [OrcidID](https://orcid.org/0000-0002-1326-6305) ) at the Laboratory for Optics and Biosciences at Ecole Polytechnique part of the Insitut Polytechnique de Paris.

The statistical analysis in R was done by Marion Aguirrebengoa of the [Big-A platform](https://cbi-toulouse.fr/fr/equipe-big-a).


Licences
========

CellProfiler and JupyterNotebooks are part of the larger python ecosystem, and as such these tools are released under the permissive BSD licence to facilitate its use; arguments for using BSD in scientific sotfware can be found for example [here](https://www.astrobetter.com/blog/2014/03/10/the-whys-and-hows-of-licensing-scientific-code/).


Funding
=======

This development was supported by:
- Agence Nationale de la Recherche (contract ANR-11-EQPX-0029 Morphoscope2) funding M-S Phan, A Chessel and D Keller
- European Research Council teloHOOK/ERC grant (714653 to L.C.) funding D Keller
- The COST Action (CA15124) - A Network of European BioImage Analysts to advance life science imaging (NEUBIAS) to D Keller to visit the CellProfiler team



Acknowledgments
=======
D Keller would like to acknowledge to Drs Beth Cimini, David Stirling and Nasim Jamali for their invaluable input for the development of the CellProfiler pipeline.


