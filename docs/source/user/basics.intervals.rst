.. _intervals:

**************
Intervals
**************

.. todo::

    - Closed on the left, open on the right, e.g. [0,1)
    - Time-like axis
    - Subset of the support, used to query a stream to get a signature or a log signature

We can create an interval to query a stream over, for example to compute a signature, in the following way. The example below is the interval [0,1), over the Reals.


::

    interval = rp.RealInterval(0, 1)