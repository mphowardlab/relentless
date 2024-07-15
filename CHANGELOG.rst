Release notes
=============

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
