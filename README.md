# DEM2mosaic

Turns a DEM TIFF and an OpenSfM reconstruction file into an orthomosaic.

## Building

This program runs on Linux. See [Dockerfile](./Dockerfile) for an example build pipeline.

## Usage

```sh
dem2mosaic <reconstruction file (.json)> <dem file (.tiff)> <georef file> <output directory>
```

Example:

```sh
dem2mosaic flight5/reconstruction.json  flight5/mesh_dsm.tif flight5/coords.txt 
```

In this example images should be located in `flight5/images`.

## License

This software is provided by Lab 308 under the [AGPL version 3](https://www.gnu.org/licenses/agpl-3.0.html).

## Acknowledgements

This software was inspired by  the paper [*Let There Be Color! Large-Scale Texturing of 3D Reconstructions*](https://download.hrz.tu-darmstadt.de/pub/FB20/GCC/paper/Waechter-2014-LTB.pdf) by Michael Waechter, Nils Moehrle, and Michael Goesele and its implementation in [MVS-Texturing](https://github.com/nmoehrle/mvs-texturing).

This software depends on [Eigen](http://eigen.tuxfamily.org) and [mapMAP](http://www.gcc.tu-darmstadt.de/home/proj/mapmap).

The inputs for this program can be created by [OpenDroneMap](https://github.com/OpenDroneMap/ODM).