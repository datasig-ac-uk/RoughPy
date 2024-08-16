.. _intervals:

**************
Intervals
**************

^^^^^^^^^^^^^^^^^^^^^
What are intervals
^^^^^^^^^^^^^^^^^^^^^

Intervals are used to query a stream to get a signature or a log signature. They are a time-like axis which is closed on the left, open on the right, e.g. [0,1).

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
How do intervals fit into RoughPy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

RoughPy is very careful in how it works with intervals.

One design goal is that it should be able to handle jumps in the underlying signal that occur at particular times, including the beginning or end of the interval, and still guarantee that if you combine the signature over adjacent interval, you always get the signature over the entire interval (i.e. Chen's identity is satisfied).
This implies that there has to be a decision about whether data at the exact beginning or exact end of the interval is included.
The convention in RoughPy are that we use clopen intervals, and that data at beginning of the interval is seen, and data at the end of the interval is seen in the next interval.

A second design goal is that the code should be efficient, and so the internal representation of a stream involves caching the signature over dyadic intervals of different resolutions.
Recovering the signature over any interval using the cache has logarithmic complexity (using at most 2n tensor multiplications, when n is the internal resolution of the stream).
Resolution refers to the length of the finest granularity at which we will store information about the underlying data.

Any event occurs within one of these finest granularity intervals, multiple events occur within the same interval resolve to a more complex log-signature which correctly reflects the time sequence of the events within this grain of time.
However, no query of the stream is allowed to see finer resolution than the internal resolution of the stream, it is only allowed to access the information over intervals that are a union of these finest resolution granular intervals.
For this reason, a query over any interval is replaced by a query is replaced by a query over an interval whose endpoints have been shifted to be consistent with the granular resolution, obtained by rounding these points to the contained end-point of the unique clopen granular interval containing this point.
In particular, if both the left-hand and right-hand ends of the interval are contained in the clopen granular interval, we round the interval to the empty interval.
Specifying a resolution of 32 or 64 equates to using integer arithmetic.

::

                                   0.3                               0.6
    --------------------------------[---------------------------------)-----------------  original interval
    [------------)[------------)[------------)[------------)[------------)[------------)  granular dyadic dissection
    0          0.125          0.25         0.375          0.5          0.625         0.75

The lower bound of the query interval (0.3) lies in the granular dyadic interval [0.25, 0.375) so is rounded to the contained end-point 0.25.
The upper bound of the query interval (0.6) lies in the granular dyadic interval [0.5, 0.625) so is rounded to the contained end-point 0.5.
This rounding means that we can ensure the the signatures over adjacent intervals always concatenate to give the correct signature over the union of these intervals.


^^^^^^^^^^^^^^^^^^^^^^^^^^
How to work with intervals
^^^^^^^^^^^^^^^^^^^^^^^^^^

We can create an interval to query a stream over, for example to compute a signature, in the following way. The example below is the interval [0,1), over the Reals.


::

    >>> interval = rp.RealInterval(0, 1)
    >>> print(f"{interval=!s}")
    interval=[0, 1)


There are other types of intervals, such as Dyadic intervals:

::

    >>> d = rp.DyadicInterval(17, 3)
    >>> print(f"{d=!s}")
    d=[2.125, 2.25)

which can be used to compute a signature or a log signature of a stream over, just as easily.
We can also use partitions:

::

    >>> p = rp.Partition(rp.RealInterval(0.5, 1.6), [0.9, 1.3])
    >>> print(f"{p=!s}")
    p=[0.5, 0.9)[0.9, 1.3)[1.3, 1.6)

which by default have the join as open on the left, closed on the right, as above.

We can also define intervals using an infimum and a suprimim. Here we define a real interval with the infimum set as 0.0 and the suprimum set as 1.0.

::

    >>> ivl = rp.RealInterval(0.0, 1.0, rp.IntervalType.Clopen)
    >>> print(f"{ivl=!s}")
    ivl=[0, 1)

.. note::

    Clopen is currently the only supported interval type.

