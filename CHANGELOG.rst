Release notes
=============

0.2.0 - 2024-07-26
------------------

*Added*

* NumPy 2 is supported (#253).
* Python 3.12 is supported (#256).
* LAMMPS 2Aug2023 is tested (#256).
* HOOMD 4 is supported (#257).

*Fixed*

* Scale factors for steepest descent methods and line search to work as
  documented (#252). The steepest descent was previously off by a square root,
  and the line search did not use them.
* README will display correctly on PyPI (#254).
* Potential name clashes when loading from JSON files are resolved more robustly
  (#255). Names can now be overridden or ignored. This fixes rare issues
  iteratively loading files.

0.1.1 - 2024-05-02
------------------

*Added*

* freud 3 is now supported (#239).

*Fixed*

* :math:`L_z` for 2D boxes is set correctly in both HOOMD 2 and HOOMD 3 (#232).
* For mixtures, :math:`g_{ij}` is now calculated using both type *i* and type
  *j* as centers. In general, the RDF now uses the standard statistical mechanics
  definition, i.e., for an ideal gas :math:`g(r) = 1-1/N` and **not** 1 (#246).

*Changed*

* GitHub Actions versions are now current and will be automatically updated
  moving forward (#240).

0.1.0 - 2023-11-15
------------------
Initial release.
