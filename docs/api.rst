API reference
=============

xfvcom.plot.core
----------------

.. automodule:: xfvcom.plot.core
   :members:
   :undoc-members:
   :show-inheritance:

Forcing visualisation
---------------------

Reusable, region-agnostic engines that visualise the boundary condition /
forcing FVCOM consumes, read straight from the input NetCDFs (no model
re-run), so figures regenerate after every BC rebuild. ``river_forcing``
and ``met_forcing`` are the time-series / spatial-map engines;
``freshwater_map`` draws where the freshwater sources sit. Thin,
region-specific drivers (paths, station registry, label CSV) live in the
consuming project (e.g. TB-FVCOM ``hydro/analysis/forcing_inputs`` and
``plot_freshwater_nodes.py``).

xfvcom.plot.river_forcing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: xfvcom.plot.river_forcing
   :members:
   :show-inheritance:

xfvcom.plot.met_forcing
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: xfvcom.plot.met_forcing
   :members:
   :show-inheritance:

xfvcom.plot.freshwater_map
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: xfvcom.plot.freshwater_map
   :members:
   :show-inheritance:

