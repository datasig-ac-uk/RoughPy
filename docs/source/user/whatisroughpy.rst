.. _whatisroughpy:

****************
What is RoughPy?
****************

RoughPy is a library for working with and analysing streamed data through the lens of rough paths and signature methods.

Most data we work with is not nice, organised, tabular data. Our data is usually messy, sparse, and unordered. RoughPy helps us work with and analyse such data.
RoughPy has no independent time steps. We can use intelligent caching so that we do not need to recompute features over and over again as we stream data. This makes our investigations easier, as can be seen in :cite:t:`kidger_neural_2020`.

RoughPy's major components are an algebra library for Lie algebras, free tensors, shuffle tensors, intervals and a library for working with streaming data. We can work with data of many file formats, such as sound files and .csvs.

.. video:: ../_static/terry.mp4
    :width: 700
    :autoplay:
    :muted:

|

References
==========

.. bibliography::

.. todo::

    Ideas for this page

    * Maud Right adjoint example
    * Signature kernels
    * Picture at bottom, paracetamol?
    * Financial volatility data, line updating, calculating signatures side by side