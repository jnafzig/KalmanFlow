---
layout: default
title: Kalman Filter
---

This code is an implementation of the Kalman Filter described [here][kalman-wiki].  The results are below:

![position and velocity plots](simple.png)

Here the solid black line is the ground truth, the shaded region is the Kalman Filter estimate and the red dots show the observations.

The filter is implemented in python using the TensorFlow framework.  Below is the computational graph:

![TensorFlow Graph](kalmangraph.png)

[kalman-wiki]: https://en.wikipedia.org/wiki/Kalman_filter#Example_application,_technical
