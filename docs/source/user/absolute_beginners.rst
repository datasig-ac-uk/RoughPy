
******************************************
RoughPy: the absolute basics for beginners
******************************************

RoughPy is a package for viewing streams of data through the lens of rough paths and using the tools and techniques of signatures.
In keeping with the style used by NumPy and other scientific Python libraries, we usually import RoughPy with the short name ``rp``::

    import roughpy as rp

In order to showcase some of the capabilities of RoughPy, we're going to walk through a simple example, where we use streams and signatures to understand the English language.
In order to do this, we need to understand the algebraic objects that will be involved.
We can think of the english language as a collection of words, each of which is made up of a sequence of letters.
If we treat each word as a stream of letters, we can regard a word as a path in a 26-dimensional space, where each letter is assigned a unique dimension.
Here, each time a letter occurs, we increment 1 unit in the direction that corresponds to the letter.
For example, take the word "stream".
We can construct the sequence of changes/increments (as numpy arrays) using the index of each character, which for lower case letters in ascii start at index 97, as follows::

    import numpy as np

    word = "stream"
    for i, letter in enumerate(word):
        vec = np.zeros(26, dtype=np.int8)
        vec[ord(letter) - 97] = 1
        print(i, letter, vec)

The ``ord`` function returns the numerical representation of a single character according to the ascii scheme. Lower case letters (a-z) are numbered between 97 (a) and 122 (z).
We use this index, minus 97 so that the letter a has index 0, to place a 1 in the correct place of the newly constructed vector.
The result looks like this::

    0 s [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
    1 t [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0]
    2 r [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]
    3 e [0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    4 a [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    5 m [0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0]

The array shown here constitutes the increments of our stream.
Before we can construct a RoughPy ``Stream`` object, we need to fix the *width*, *depth*, and *coefficient type*.
The *width* refers to the dimension of the vector space in which our stream takes its values.
In this case, the width should be 26 (there are 26 letters in the English alphabet).
The *depth* refers to the level at which signatures and log-signatures should be truncated.
We will revisit this later.
The *coefficient type* describes the type of scalars that are used in calculations. Usually this will be some kind of floating point number such as double precision (``rp.DPReal``) or single precision (``rp.SPReal``), but RoughPy also supports arbitrary precision rational arithmetic (``rp.Rational``), which is what we will use in this example.
These three parameters are usually collected together into an ``AlgebraContext``, which is constructed as follows::

    CTX = rp.get_context(width=26, depth=3, coeffs=rp.Rational)

We then use this to construct a stream object.

Constructing a stream
---------------------
There are a number of ways to construct a stream in RoughPy, depending on the nature of the underlying data and the type of stream we need.
For the word (increment) streams described above, we should use the ``LieIncrementStream``, and the ``from_increments`` constructor function.
Since we're going to use this many times later, let's write a small function that takes an arbitrary word (consisting of lower case letters) and constructs a stream object using the context we obtained at the end of the previous section::

    def word_to_stream(word):
        increment_array = np.zeros((len(word), 26), dtype=np.int8)
        for i, letter in enumerate(word):
            letter_idx = ord(letter)
            assert 97 <= letter_idx <= 122, f"expected lower case letter, got {letter}"
            increment_array[i, letter_idx - 97] = 1

        return rp.LieIncrementStream.from_increments(increment_array, resolution=2, ctx=CTX)

Here, we collect all the vectors together into a single array with ``len(word)`` rows and 26 columns using the same logic as above.
(We've also included an assertion that the characters that appear are lower-case letters for safety.)
The ``LieIncrementStream.from_increments`` function is provided with the increment data, and the context object using the ``ctx`` keyword argument.
We also provide a ``resolution``, which is used internally during computations.
A detailed discussion about this parameter can be found `here <https://github.com/datasig-ac-uk/RoughPy/issues/53>`_.


.. note::
    There are two related concepts that describe the type of data that can be used to construct a stream. These are *values* and *increments*.
    With increment data, we provide a list of changes that occur as the the data evolves whereas with value data we describe the values that appear.
    In the example above, each occurrence of a letter is an increment.
    Value data could instead be constructed by taking the cumulative sum of the increment data.
    Conversely, increment data can be obtained from value data by taking the row-to-row differences.


We can now construct a stream representation of our word and compute its signature as follows::

    stream = word_to_stream(word)
    signature = stream.signature()
    print(signature)

Without any arguments, the ``signature`` method computes the signature over the whole domain on which the stream is defined, up to the depth specified when we created the context (3).
(In this case, since we did not specify parameter values, the increments occur at the positive integer parameter values 0.0, 1.0, 2.0, and so on.)
The signature of the word ``stream`` should look as follows::

    { 1() 1(1) 1(5) 1(13) 1(18) 1(19) 1(20)
      1/2(1,1) 1(1,13) 1(5,1) 1/2(5,5) 1(5,13) 1/2(13,13) 1(18,1)
      1(18,5) 1(18,13) 1/2(18,18) 1(19,1) 1(19,5) 1(19,13) 1(19,18)
      1/2(19,19) 1(19,20) 1(20,1) 1(20,5) 1(20,13) 1(20,18) 1/2(20,20)
      1/6(1,1,1) 1/2(1,1,13) 1/2(1,13,13) 1/2(5,1,1) 1(5,1,13) 1/2(5,5,1)
      1/6(5,5,5) 1/2(5,5,13) 1/2(5,13,13) 1/6(13,13,13) 1/2(18,1,1)
      1(18,1,13) 1(18,5,1) 1/2(18,5,5) 1(18,5,13) 1/2(18,13,13) 1/2(18,18,1)
      1/2(18,18,5) 1/2(18,18,13) 1/6(18,18,18) 1/2(19,1,1) 1(19,1,13)
      1(19,5,1) 1/2(19,5,5) 1(19,5,13) 1/2(19,13,13) 1(19,18,1) 1(19,18,5)
      1(19,18,13) 1/2(19,18,18) 1/2(19,19,1) 1/2(19,19,5) 1/2(19,19,13)
      1/2(19,19,18) 1/6(19,19,19) 1/2(19,19,20) 1(19,20,1) 1(19,20,5)
      1(19,20,13) 1(19,20,18) 1/2(19,20,20) 1/2(20,1,1) 1(20,1,13) 1(20,5,1)
      1/2(20,5,5) 1(20,5,13) 1/2(20,13,13) 1(20,18,1) 1(20,18,5) 1(20,18,13)
      1/2(20,18,18) 1/2(20,20,1) 1/2(20,20,5) 1/2(20,20,13) 1/2(20,20,18)
      1/6(20,20,20) }

The signature is an element of the `free tensor algebra <https://en.wikipedia.org/wiki/Tensor_algebra>`_ and is an abstract description of the stream.
The format of the entries is "``coefficient(tensor_word)``", where ``coefficient`` is a rational number, and ``tensor_word`` is a comma-separated list of "letters" - numbers that label the underlying dimensions.
(Here the tensor letter 1 corresponds to ``a``, 2 to ``b``, and so on.)
The first term of the signature is always ``1()``, where ``()`` is the empty-word.
The next few terms are the tensor words of length 1, which are also shown on the first line of the output.
In this context, these terms simply count the number of each letter that that appears.
The word we used was "stream" so, in total, there is 1 'a' (1), 1 'e' (5), 1 'm' (13), 1 'r' (18), 1 's' (19), and 1 't' (20).
The terms that follow correspond to the tensor words of length 2 and length 3.
The *depth* parameter that we specified earlier refers to the maximum length of tensor word that can appear in the signature.
The tensor words of length 2 capture the order in which each pair of letters occur.
For instance, the term "1(18,5)" indicates that the letter 'r' (18) appears before the letter 'e' (5).
However, note that this is not quite a perfect interpretation since the term 1/2(1,1) is also present in the output.
In fact, terms like (1,1) don't add any additional information - they are redundant terms. A ``tensor_word`` of length n formed by a repeat of some letter ``i`` is equal to the increment in the ith dimension raised to the power of n and divided by n!.

Log-signatures
--------------
The signature is a relatively large object and, as we have seen, contains a great deal of redundancy.
However, there is a remedy.
We can instead consider the *log-signature* of the stream, which contains the same information as the signature but without any of the redundancy.
The log-signature of a stream is a member of the `free Lie algebra <https://en.wikipedia.org/wiki/Free_Lie_algebra>`_, and is a different way of abstractly representing the stream.

.. note::

    While the signature and log-signature fundamentally represent the same information, their mathematical properties are quite different.
    A fundamental theorem in the literature shows that all continuous functions on the original path can be approximated as linear functionals (shuffle tensors) on the signature.
    The same simple approximation property is not available for log-signatures.
    This means that, for some applications, the signature is a more appropriate representation while in others the log-signature is better.

To compute the log-signature we use the ``log_signature`` method on the stream object::

    log_signature = stream.log_signature()
    print(log_signature)

The first few terms of the printed result will be as follows::

    1(1) 1(5) 1(13) 1(18) 1(19) 1(20) -1/2([1,5]) 1/2([1,13])

The first 6 terms here appear exactly as in the the signature.
However, the terms that follow correspond to the *Hall words*, which form a basis for the free Lie algebra.
These terms encode the order of "letters" in all possible pairs, where a negative coefficient means the order shown in the brackets should be flipped.
For instance, ``-1/2([1,5])`` indicates that 'e' (5) appears before 'a' (1).


Scaling up: classifying anagrams
-----------------------------------------------------
Let's see how to use signatures and log-signatures on a larger scale to classify all the words in the standard word list installed on Linux systems (``/usr/share/dict/words``).
We'll start with the level 1 log-signatures.
Here we'll see that all the words that have the same log-signature of depth 1 are anagrams of one another.
Obviously, there are many anagrams so the depth 1 log-signatures are not sufficient to uniquely identify all words.
However, we do not need to increase the depth much further before we can completely identify all words using their log-signature.
Let's see how to do this.

First let's extract a list of words from the dictionary file.
We have to filter out words that contain apostrophes and words of length 1, make lowercase, and remove any duplicates::

    with open("/usr/share/dict/words", "rt") as fd:
        words = {word.lower() for line in fd
                 if len(word := line.strip().replace('-', '')) > 1
                 if word.isalpha() and word.isascii()}

    print(f"There are {len(words)} to process")

This will load a list of 87950 words without duplicates, and all lower case, containing only words with length at least 2.
(The whole file contains approximate 124000 words in total.
Your list may vary in size depending on the specific version used.)
The next thing we need to do is process all of these words into RoughPy streams.
We can use the ``word_to_stream`` function we defined above for this task::

    word_streams = {word: word_to_stream(word) for word in words}

Next, we can compute log-signatures for all the streams in this collection.
Ideally, we'd like to collect together words with the same signature in a dictionary-like construction.
Unfortunately, for technical reasons, RoughPy Lie objects cannot be the key in a dictionary so we have to turn the log-signatures into strings, which can serve as the key for a dictionary.
To speed things up, we're going to use a thread pool to compute log-signatures in parallel::

    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial
    from time import time

    def compute(word_stream, *, depth):
        """Helper function for to get (key, word) results"""
        return str(word_stream[1].log_signature(depth=depth)), word_stream[0]

    anagrams = defaultdict(list)
    start = time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for key, word in pool.map(partial(compute, depth=1), word_streams.items()):
            anagrams[key].append(word)
    elapsed = time() - start

    print(f"Computation took {elapsed} seconds")

After completion - which takes around 65 seconds on this machine - we have a dictionary whose keys are (stringified) log-signatures to depth 1 and corresponding entries are lists of words with that log-signature.
(See the `documentation <https://docs.python.org/3/library/collections.html#collections.defaultdict>`_ for ``defaultdict`` for more details.)

Any entry in the dictionary where the list contains only one word means that this word is completely described only by number of each letter that appear therein.
As we shall see momentarily, this accounts for the vast majority of the words in our list.
Any remaining entries correspond to the different classes of anagrams that exist.

.. note::
    Notice that we did not need to fully specify a new context or reconstruct the stream in order to do these computations with a different depth.
    The ``signature`` and ``log_signature`` methods can take an optional ``depth`` keyword argument that instructs the stream to do the computation at a higher or lower depth than previously specified.

Let's filter our list of streams by removing any word that is completely characterised by its first level log-signature (the words that have no anagram), and at the same time print all the words which have at least 7 anagrams::

    for key, words in anagrams.items():
        if len(words) == 1:
            word_streams.pop(words[0])
        if len(words) > 6:
            print(f"{key:<40}", *words)

This should print the following::

    { 1(1) 1(5) 1(12) 1(16) 1(19) 1(20) }    pleats pastel palest plates petals septal staple
    { 1(1) 1(5) 1(12) 1(19) }                sale lase elsa lesa seal leas ales
    { 1(1) 1(3) 1(5) 1(16) 1(18) 1(19) }     pacers crapes spacer scrape capers parsec casper recaps
    { 1(1) 1(3) 1(5) 1(18) 1(19) 1(20) }     carets recast caster crates reacts caters traces
    { 1(1) 1(5) 1(12) 1(19) 1(20) }          teals least tales slate tesla stael steal stale
    { 1(1) 1(5) 1(16) 1(18) 1(19) }          pears rapes parse spear reaps spare pares
    { 1(1) 1(5) 1(18) 1(19) 1(20) }          aster stare rates taser tears resat treas tares

After filtering, we are left with 13160 words that have at least one anagram (around 15% of all words).

Increasing the depth
++++++++++++++++++++
The majority of the words in the list are easily distinguished by the depth 1 log-signature.
However, a fair number remain that cannot be distinguished.
Now we can repeat the computations above, but instead using the depth 2 log-signatures.
This should give us more distinguishing power, and we will see that this leaves only a very small number of words that cannot be distinguished::

    anagrams_2 = defaultdict(list)
    start = time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for key, word in pool.map(partial(compute, depth=2), word_streams.items()):
            anagrams_2[key].append(word)
    elapsed = time() - start

    print(f"Computation took {elapsed} seconds")


    for key, words in anagrams_2.items():
        if len(words) == 1:
            word_streams.pop(words[0])
        else:
            print(f"{key:<40}", *words)

    print(f"There are {len(word_streams)} words whose level 3 signatures are necessary")

After about 70 seconds of computation, the computation completes and once again we print the classes of words that have size greater than 1.
We are left with just four words (two pairs)::

    { 2(15) 2(20) }                          otto toot
    { 2(1) 2(14) }                           naan anna

For these words, we would have to increase the depth once again to 3 in order to fully distinguish them.

So what exactly have we identified in this latest round of computations?
Clearly the words in each category are anagrams of one another.
But the level 2 terms of the log-signature are also identifying the order in which each pair of letters occur.
So for two words to have equal log-signatures up to depth 2, they must be anagrams and all pairs of letters must appear in the same order in both words.
For instance, in both of the words from the first set displayed above, a 't' appears before an 'o', and an 'o' before a 't' (and an 'o' before an 'o' and a 't' before a 't', but these are somehow trivially satisfied).
This fact is perhaps more obvious if we look at the signature for the first category::

    { 1() 2(15) 2(20) 2(15,15) 2(15,20) 2(20,15) 2(20,20) }

Here we can see that all the second order terms are equal.
The fact that 'o' appears before 't' "cancels out" the fact that 't' appears before 'o' when we look at the log signature.

.. note::
    The time taken to compute the log-signatures for all the words that have at least one anagram is large compared to the much larger collection of all words.
    This is expected.
    The computation of a log-signature to depth 1 has complexity proportional to the width, which is 26.
    However, the computation of log-signatures to depth 2 has complexity proportional to width squared.
    This is due to the fact that size of the signature grows geometrically with depth.
    Log-signatures are smaller objects than signatures, but their computational complexity is tied to that of the signature.

It is a little surprising that depth 2 is sufficient to distinguish such a large proportion of the words in the original list.
In general, we might expect that the depth required to distinguish words would be related to the maximum length of the words.
The "rules" of English spelling are doing quite a lot of work here.

To perform an extra check, we can compute the log-signature to depth 3 for the words "toot" and "otto".
For "toot", we get that the log-signature is

::

    { 2(15) 2(20) 2/3([15,[15,20]]) 1/3([20,[15,20]]) }

but the log-signature for "otto" is

::

    { 2(15) 2(20) -1/3([15,[15,20]]) -2/3([20,[15,20]]) }

which are not equal.
The same is true for "naan" and "anna", since the pattern of letters is identical.
Only the non-zero terms are printed here. There are no (non-zero) terms of degree 2 in either of these log signatures.
