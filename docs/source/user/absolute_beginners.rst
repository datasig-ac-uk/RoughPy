
******************************************
RoughPy: the absolute basics for beginners
******************************************

.. warning::
    This page is under construction. Updates coming soon!

RoughPy is very careful in how it works with intervals.
One design goal is that it should be able to handle jumps in the underlying signal that occur at particular times, including the beginning or end of the interval, and still guarantee that if you combine the signature over adjacent interval, you always get the signature over the entire interval.
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