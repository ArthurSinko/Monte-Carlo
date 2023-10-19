# Monte-Carlo
Very brief intro to Monte Carlo 

## History of invention

Blah-blah-blah

## Idea behind

Assume you have a black box that can generate as many realisations from an unknown random variable as you want. T

# Issue

[local_gradient_for_argument](https://github.com/pim-book/programmers-introduction-to-mathematics/blob/b8867dd443f2e4350e8b257d22bdc95de2c008d5/neural_network/neural_network.py#L125) might read a corrupted float value from the successor.

Say we have the computation graph:

\`\`\`text
    h
    |
    |
    f
\`\`\`

- $h$ has parameters $w$
- $f$ has parameters $v$
- $h : \mathbb{R}^m \to \mathbb{R}$
- $f : \mathbb{R}^n \to \mathbb{R}$
- $h$ is the successor of $f$
- $E$ is the graph
- $\frac{\partial E}{\partial w} = \frac{\partial E}{\partial h} \cdot \frac{\partial h}{\partial w}$
- $\frac{\partial E}{\partial f} = \frac{\partial E}{\partial h} \cdot \frac{\partial h}{\partial f}$
- $\frac{\partial E}{\partial v} = \frac{\partial E}{\partial f} \cdot \frac{\partial f}{\partial v}$
- when doing backpropagation, the steps will be
  1. $h$ computes $\frac{\partial E}{\partial w}$ and caches $\frac{\partial E}{\partial h}$ and $\frac{\partial h}{\partial w}$
  1. $h$ updates $w$ to $w'$
  1. $f$ computes $\frac{\partial E}{\partial f}$ and $\frac{\partial h}{\partial f}$ is cached
     1. $\frac{\partial h}{\partial f}$ is not yet in cache, so $h$ will have to compute it now
     1. $\frac{\partial h}{\partial f}$ is computed based on the new parameter $w'$
        - **This is the problem!**
        - $\frac{\partial h}{\partial f}$ is **corrupted**
     1. $\frac{\partial h}{\partial f}$ is in cache now
     1. $\frac{\partial E}{\partial f}$ is computed by looking both $\frac{\partial E}{\partial h}$ and $\frac{\partial h}{\partial f}$ in cache
     1. $\frac{\partial E}{\partial f}$ is in cache now
  1. $f$ computes $\frac{\partial f}{\partial v}$ and caches it
  1. $f$ computes $\frac{\partial E}{\partial v}$ with $\frac{\partial f}{\partial v}$ and the **corrupted** $\frac{\partial h}{\partial f}$
  1. $f$ updates $v$ based on the **corrupted** $\frac{\partial E}{\partial v}$

# Solutions

I can come up with two solutions

- compute `local_gradient` ($\frac{\partial h}{\partial f}$) at the beginning of [do_gradient_descent_step](https://github.com/pim-book/programmers-introduction-to-mathematics/blob/b8867dd443f2e4350e8b257d22bdc95de2c008d5/neural_network/neural_network.py#L107) before parameters ($w$) is modified
- the successor $h$ distributes `local_gradient` ($\frac{\partial h}{\partial f}$) and `global_gradient` ($\frac{\partial E}{\partial h}$) to $f$ before parameters ($w$) is modified
  - I have also coded a neural network based on yours, the distribution happens at [distribute_global_gradient_entries_to_operands](https://github.com/Banyc/neural_network/blob/c6970d2712c8c01286b0a10434723a073fb99ad7/src/nodes/node.rs#L116)
