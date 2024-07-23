.. _streams:

**************
Streams
**************

^^^^^^^^^^^^^^^^^^^^^
What are streams
^^^^^^^^^^^^^^^^^^^^^

A stream means an object that provides the signature or log-signature over any interval. For more information on streams, see :cite:t:`lyons_signature_2024` and :cite:t:`lyons_differential_2007`.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
How do streams fit into RoughPy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Streams are parametrised sequential data viewed via Rough Path theory as a rough path.

^^^^^^^^^^^^^^^^^^^^^^^^
How to work with streams
^^^^^^^^^^^^^^^^^^^^^^^^

To create a stream, you need **data**.
For example, we can create data using a linearly spaced time array.

::

    >>> times = np.linspace(0.1,1,10)
    >>> times2 = 2*times
    >>> data = np.concatenate([times.reshape(-1,1),
                               times2.reshape(-1,1),
                               (times*d2t - 2*times*dt).reshape(-1,1)], axis=1)

To create a stream, we should also provide a **context**. For more information, see :doc:`fundamentals.contexts`.

For example, we can create the following context, with width 2, depth 6, and Real coefficients. The depth specification for constructing the stream is not the depth of the data, but it is the depth of the stream. The depth of the data must be less than or equal to the depth of the stream.

::

    >>> context = roughpy.get_context(width=2, depth=6, coeffs=roughpy.DPReal)

You can then create a **stream** using your chosen **data** and **context** with **Lie** increments.

::

    >>> stream = roughpy.LieIncrementStream.from_increments(data, indices=times, ctx=context)

You can also compute and display the signature and log signature of any stream.

::

    >>> sig = stream.signature(interval)
    >>> print(f"{sig=!s}")
    sig={ 1() 4.5(1) 9(2) 10.125(1,1) 20.25(1,2) 20.25(2,1) 40.5(2,2) 15.1875(1,1,1) 30.375(1,1,2) 30.375(1,2,1) 60.75(1,2,2) 30.375(2,1,1) 60.75(2,1,2) 60.75(2,2,1) 121.5(2,2,2) 17.0859(1,1,1,1) 34.1719(1,1,1,2) 34.1719(1,1,2,1) 68.3438(1,1,2,2) 34.1719(1,2,1,1) 68.3438(1,2,1,2) 68.3438(1,2,2,1) 136.688(1,2,2,2) 34.1719(2,1,1,1) 68.3438(2,1,1,2) 68.3438(2,1,2,1) 136.688(2,1,2,2) 68.3438(2,2,1,1) 136.688(2,2,1,2) 136.688(2,2,2,1) 273.375(2,2,2,2) 15.3773(1,1,1,1,1) 30.7547(1,1,1,1,2) 30.7547(1,1,1,2,1) 61.5094(1,1,1,2,2) 30.7547(1,1,2,1,1) 61.5094(1,1,2,1,2) 61.5094(1,1,2,2,1) 123.019(1,1,2,2,2) 30.7547(1,2,1,1,1) 61.5094(1,2,1,1,2) 61.5094(1,2,1,2,1) 123.019(1,2,1,2,2) 61.5094(1,2,2,1,1) 123.019(1,2,2,1,2) 123.019(1,2,2,2,1) 246.038(1,2,2,2,2) 30.7547(2,1,1,1,1) 61.5094(2,1,1,1,2) 61.5094(2,1,1,2,1) 123.019(2,1,1,2,2) 61.5094(2,1,2,1,1) 123.019(2,1,2,1,2) 123.019(2,1,2,2,1) 246.038(2,1,2,2,2) 61.5094(2,2,1,1,1) 123.019(2,2,1,1,2) 123.019(2,2,1,2,1) 246.038(2,2,1,2,2) 123.019(2,2,2,1,1) 246.038(2,2,2,1,2) 246.038(2,2,2,2,1) 492.075(2,2,2,2,2) 11.533(1,1,1,1,1,1) 23.066(1,1,1,1,1,2) 23.066(1,1,1,1,2,1) 46.132(1,1,1,1,2,2) 23.066(1,1,1,2,1,1) 46.132(1,1,1,2,1,2) 46.132(1,1,1,2,2,1) 92.2641(1,1,1,2,2,2) 23.066(1,1,2,1,1,1) 46.132(1,1,2,1,1,2) 46.132(1,1,2,1,2,1) 92.2641(1,1,2,1,2,2) 46.132(1,1,2,2,1,1) 92.2641(1,1,2,2,1,2) 92.2641(1,1,2,2,2,1) 184.528(1,1,2,2,2,2) 23.066(1,2,1,1,1,1) 46.132(1,2,1,1,1,2) 46.132(1,2,1,1,2,1) 92.2641(1,2,1,1,2,2) 46.132(1,2,1,2,1,1) 92.2641(1,2,1,2,1,2) 92.2641(1,2,1,2,2,1) 184.528(1,2,1,2,2,2) 46.132(1,2,2,1,1,1) 92.2641(1,2,2,1,1,2) 92.2641(1,2,2,1,2,1) 184.528(1,2,2,1,2,2) 92.2641(1,2,2,2,1,1) 184.528(1,2,2,2,1,2) 184.528(1,2,2,2,2,1) 369.056(1,2,2,2,2,2) 23.066(2,1,1,1,1,1) 46.132(2,1,1,1,1,2) 46.132(2,1,1,1,2,1) 92.2641(2,1,1,1,2,2) 46.132(2,1,1,2,1,1) 92.2641(2,1,1,2,1,2) 92.2641(2,1,1,2,2,1) 184.528(2,1,1,2,2,2) 46.132(2,1,2,1,1,1) 92.2641(2,1,2,1,1,2) 92.2641(2,1,2,1,2,1) 184.528(2,1,2,1,2,2) 92.2641(2,1,2,2,1,1) 184.528(2,1,2,2,1,2) 184.528(2,1,2,2,2,1) 369.056(2,1,2,2,2,2) 46.132(2,2,1,1,1,1) 92.2641(2,2,1,1,1,2) 92.2641(2,2,1,1,2,1) 184.528(2,2,1,1,2,2) 92.2641(2,2,1,2,1,1) 184.528(2,2,1,2,1,2) 184.528(2,2,1,2,2,1) 369.056(2,2,1,2,2,2) 92.2641(2,2,2,1,1,1) 184.528(2,2,2,1,1,2) 184.528(2,2,2,1,2,1) 369.056(2,2,2,1,2,2) 184.528(2,2,2,2,1,1) 369.056(2,2,2,2,1,2) 369.056(2,2,2,2,2,1) 738.113(2,2,2,2,2,2) }

    >>> logsig = stream.log_signature(interval)
    >>> print(f"{logsig=!s}")
    logsig={ 4.5(1) 9(2) }

The above stream was constructed with Lie increments. You can construct streams in many ways. The six types are:

#. Lie increment streams

    ::

        # Using Lie increments data as above
        >>> lie_increment_stream = roughpy.LieIncrementStream.from_increments(data, indices=times, ctx=context)

#. Tick streams

    ::

       # create tick stream data
       >>> data = { 1.0: [("first", "increment", 1.0),("second", "increment", 2.0)],
                2.0: [("first", "increment", 1.0)]
                }

       # construct stream
       >>> tick_stream = TickStream.from_data(data, width=2, depth=2, dtype=DPReal)

#. Brownian streams

    ::

        # Generating on demand from a source of randomness with normal increments that approximate Brownian motion
        >>> brownian_stream = BrownianStream.with_generator(width=2, depth=2, dtype=DPReal)

#. Piecewize abelian streams

    ::

        # create piecewise lie data
        >>> piecewise_intervals = [RealInterval(float(i), float(i + 1)) for i in range(5)]

        >>> piecewise_lie_data = [
                (interval,
                brownian_stream.log_signature(interval))
                for interval in piecewise_intervals
            ]

        # construct stream
        >>> piecewise_abelian_stream = PiecewiseAbelianStream.construct(piecewise_lie_data, width=2, depth=2, dtype=DPReal)

#. Function streams

    ::

        # create a function to generate a stream from
        >>> def func(t, ctx):
        ...     return Lie(np.array([t, 2*t]), ctx=ctx)

        #construct stream
        >>> function_stream = rp.FunctionStream.from_function(func, width=2, depth=2, dtype=rp.DPReal)

#. External source data (two kinds)

    ::

        # create a stream from an external source
        # here we use a sound file, but other formats are supported
        >>> roughpy.ExternalDataStream.from_uri("/path/to/sound_file.mp3", depth=2)


You can also plot your streams. For example, here is a stream from electricity data, which can be downloaded `here <https://data.ukedc.rl.ac.uk/browse/edc/efficiency/residential/EnergyConsumption/Domestic/UK-DALE-2017/ReadMe_DALE-2017.html#HighSpeed>`_.

.. figure:: ../_static/1.png

    Fig 1. Raw signal over a time interval at the specified resolution. Accessing all the level one increments at the fine resolution and using these values to plot the two dimensional over the interval.

..
    .. figure:: ../_static/3.png

        Fig 2. Applying UMAP to learn the manifold embedding that captures the log signature features we observe in electric data to classify cycles in an unsupervised way.

    .. figure:: ../_static/5.png

        Fig 3. Plotting the UMAP reduced data as a function of cycle index.

To see how these streams were made, queried and then plotted, see `Electricity Data Example <https://github.com/datasig-ac-uk/electricity-data-example/tree/main>`_.

^^^^^^^^^^^^^^^^^^^^^
Literature references
^^^^^^^^^^^^^^^^^^^^^

.. bibliography::
