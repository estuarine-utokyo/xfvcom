Quick usage
===========

Install
-------

.. code-block:: bash

   pip install xfvcom     # or: conda install -c conda-forge xfvcom

Plotting example
----------------

.. code-block:: python

   from xfvcom.plot.core import FvcomPlotter, FvcomPlotConfig
   plotter = FvcomPlotter(ds, FvcomPlotConfig())
   ax = plotter.plot_2d(ds["temp"].isel(time=0, siglay=0))

