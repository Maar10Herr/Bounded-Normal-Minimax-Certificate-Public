# Bounded Normal Mean Minimax Certificate

A deterministic adaptive-support computation and independent interval
certificate for one bounded-normal minimax problem.

> **Software status**
>
> Experimental research implementation and computer-assisted numerical
> certificate. This is not a general minimax solver, a formal proof-assistant
> development, or a replacement for the established bounded-normal and
> least-favourable-prior literature.

For

$$
Y\mid\theta\sim N(\theta,1),\qquad \theta\in[-m,m],
$$

under squared-error loss, the repository separates candidate discovery from
verification. A float64 solver searches for a symmetric finitely supported
prior. A materially independent verifier then reconstructs the risk directly
with 192-bit Arb balls, outward-rounded quadrature and tail bounds, and
continuous branch-and-bound.

The release contains one final certificate, for $m=4$. The verifier’s stored
claim is

$$
v^*\in[0.815088756,\,0.815088764],\qquad
U-L\le 8\times10^{-9}.
$$

The paper reports a tighter independently recomputed enclosure from the full
Arb output. The JSON values above are the deliberately rounded release contract.

**[Read the research paper](paper/Bounded_Normal_Mean_Minimax_Certificate.pdf).**

## What is certified

For a supplied finite symmetric prior $\pi$, its Bayes risk $b(\pi)$ and the
maximum risk $U(\pi)$ of its Bayes rule satisfy

$$
b(\pi)\le v^*\le U(\pi).
$$

The verifier independently encloses both endpoints. If the enclosure width is
at most $\varepsilon$, the accepted Bayes rule is certified
$\varepsilon$-minimax in value.

This does **not** prove that the discovery solver finds a certifiable prior in
finite time, establish a general complexity bound, or make the float64 sweep a
certificate. The trusted base includes CPython, python-flint/Arb, and the
verifier source.

## Architecture and trust boundary

```text
float64 discovery solver
  Gauss-Hermite evaluation
  mass/location reoptimization
  adaptive support insertion
             |
             v
       compact JSON certificate
             |
             v
independent Arb verifier
  direct Gaussian-mixture ratios
  outward-rounded Simpson bounds
  Gaussian tail enclosures
  continuous branch-and-bound on [0,m]
             |
             v
       accepted risk interval
```

The final verifier does not import either solver and does not accept a
serialized discovery branch tree. It rebuilds continuous coverage from
$[0,m]$ and rejects malformed support, mass, precision, interval, and
normalization claims.

## Reproduce the certificate

Requirements: Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```sh
uv sync --frozen
uv run python -m unittest discover -s tests -v
uv run python src/verification/verify_certificate_arb.py \
  certificates/final/m4_epsilon_1e-8_arb.json
```

The verifier exits nonzero when any structural, numerical, or final-gap check
fails. A separate corruption audit deliberately mutates the certificate:

```sh
uv run python -m scripts.run_corruption_audit \
  --certificate certificates/final/m4_epsilon_1e-8_arb.json \
  --output /tmp/bounded-normal-corruption-audit.json
```

The audit output is intentionally written outside the repository; it is not a
release artifact.

## Certificate contents

`certificates/final/m4_epsilon_1e-8_arb.json` contains only:

- the bound $m=4$;
- three nonnegative support levels and their total symmetric masses;
- precision, quadrature, tail, branch, and work-tolerance settings; and
- claimed Bayes-risk, maximum-risk, minimax-risk, and gap intervals.

It contains no samples, parameter grid, random draws, runtime database,
machine path, private hash, or discovery branch tree.

## Prior work and contribution scope

The outer exchange architecture—finite-prior optimization, global risk
separation, support insertion, pruning, and the Bayes-risk/maximum-risk stopping
bracket—is prior art from [Kempthorne (1987)](https://doi.org/10.1137/0908028).
Foundational bounded-normal results include
[Casella and Strawderman (1981)](https://doi.org/10.1214/aos/1176345527), while
[Gourdin, Jaumard, and MacGibbon (1994)](https://doi.org/10.1137/0915002)
provide important deterministic global-optimization precedent.

[Montiel Olea and Zubova (2026)](https://arxiv.org/abs/2607.05350) develop a
stochastic mirror-ascent approach with an equally spaced parameter grid and
convergence guarantees. Their bounded-normal computation motivated this
project. The present implementation addresses a narrower question: certifying
one scalar bounded-normal value deterministically after a candidate prior has
been found. It neither reproduces their larger LP/VAR application nor claims
their method is inferior.

The contribution claimed here is limited to the specialized deterministic
implementation, the compact $m=4$ artifact, and the independently executable
outward-rounded verifier. The posterior common-mean enclosure was derived in
this project, but no historical priority is claimed. General interval
arithmetic, least-favourable-prior theory, posterior moment identities, and
semiconvex branch-and-bound are not claimed as original.

The paper contains the complete 15-item bibliography and a claim-lineage table.
Citation denotes intellectual antecedence, not endorsement or co-authorship.

## Repository layout

```text
paper/                  publication PDF; LaTeX source is not distributed
certificates/final/     accepted compact m=4 certificate
src/solver/             discovery and certificate-construction implementations
src/verification/       independent Arb verifier
scripts/                corruption-audit helper
tests/                  identities, numerical bounds, and corruption tests
pyproject.toml           pinned project metadata
uv.lock                  reproducible dependency lock
```

Correspondence, prior-art extracts, exploratory certificates, research logs,
build products, working notes, and redundant wrappers are not distributed in
this repository.

## Limitations

- The final certificate covers only the scalar unit-variance model at $m=4$.
- Results for other bounds in the discovery sweep are float64 diagnostics.
- The verifier is computer-assisted and depends on its stated software trusted
  base; it is not a proof-assistant kernel.
- No finite-iteration or complexity theorem is supplied for discovery.
- Timing measurements are machine-specific and are not performance claims.
- The literature audit is source-grounded but cannot establish an exhaustive
  absence of closer antecedents.

## Citation and license

Use [`CITATION.cff`](CITATION.cff) to cite the versioned research software and
paper. The original code and certificate in this repository are released under
the [MIT License](LICENSE). The cited papers and their contents remain the work
of their respective authors and publishers.
