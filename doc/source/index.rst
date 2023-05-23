==========
relentless
==========

``relentless`` is a Python package for executing molecular simulations in larger
computational workflows. The simulation protocol is specified as a reproducible,
human-readable recipe that is run natively in popular engines such as
:class:`~relentless.simulate.LAMMPS` and :class:`~relentless.simulate.HOOMD`.
``relentless`` also has robust features for optimization with simulations,
including independent and dependent variables with automatic differentiation,
objective functions of simulations, and optimization methods. All features are
readily extensible through a clean hierarchy of abstract objects, enabling you
to quickly use your own simulation code or optimization objective.
Our initial goal was to enable optimization of the ``rel``\ ative ``ent``\ ropy
of structural coarse-graining and materials design: with ``less`` code.

.. rubric:: :doc:`guide/index`
.. grid:: 2 2 4 4

    .. grid-item-card::

        .. button-ref:: guide/install
            :expand:
            :color: secondary
            :click-parent:

    .. grid-item-card::

        .. button-ref:: guide/examples/index
            :expand:
            :color: secondary
            :click-parent:

.. rubric:: :doc:`api/index`
.. grid:: 2 2 4 4

    .. grid-item-card::

        .. button-ref:: api/model
            :expand:
            :color: primary
            :click-parent:

            Model

    .. grid-item-card::

        .. button-ref:: api/simulate
            :expand:
            :color: primary
            :click-parent:

            Simulate

    .. grid-item-card::

        .. button-ref:: api/optimize
            :expand:
            :color: primary
            :click-parent:

            Optimize

    .. grid-item-card::

        .. button-ref:: api/index
            :expand:
            :color: primary
            :click-parent:

            More

.. toctree::
    :maxdepth: 1
    :hidden:

    guide/index
    api/index

.. rubric:: Indices and tables

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
