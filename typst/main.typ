#import "@preview/touying:0.6.1": *
#import themes.university: *
#import "@preview/cetz:0.3.2"
#import "@preview/fletcher:0.5.4" as fletcher: node, edge
#import "@preview/numbly:0.1.0": numbly
#import "@preview/theorion:0.3.2": *
#import cosmos.clouds: *
#import "@preview/codly:1.2.0": *
#import "@preview/codly-languages:0.1.1": *
#show: codly-init.with()
#show: show-theorion

#codly(
  languages: (
    julia: (name: "Julia", color: rgb("#9558B2")),
  )
)

// cetz and fletcher bindings for touying
#let cetz-canvas = touying-reducer.with(reduce: cetz.canvas, cover: cetz.draw.hide.with(bounds: true))
#let fletcher-diagram = touying-reducer.with(reduce: fletcher.diagram, cover: fletcher.hide)

#show: university-theme.with(
  aspect-ratio: "16-9",
  // align: horizon,
  // config-common(handout: true),
  config-common(frozen-counters: (theorem-counter,)),  // freeze theorem counter for animation
  config-info(
    title: [A Demo of HINTS],
    // subtitle: [A Demo of HINTS],
    author: [Dinghang Tan],
    date: datetime.today(),
    // institution: [XMU],
    // logo: emoji.school,
  ),
)

#set heading(numbering: numbly("{1}.", default: "1.1"))

#title-slide()

== Outline <touying:hidden>

#components.adaptive-columns(outline(title: none, indent: 1em))
#show: magic.bibliography-as-footnote.with(bibliography("refs.bib", title: none))

= Basic Principles
== Smoothing Principles
+ A basic residual-correction method to solve $A u = f$ for a given initial guess $u_0$.
  + residual $r = f - A u_k$.
  + correction $e = B r$ with $B approx A^(-1)$.
  + $u_(k+1) = u_k + e$.
+ Amplification matrix.
  $ e_(k+1) &:= u - u_(k+1) = (I - B A) (u-u_k) = (I - B A)e_k \
    E &:= I - B A
  $
  #theorem()[
    The residual-correction shceme converges iff 
    $
      rho(I - B A) < 1.
    $
  ]

+ Jacobi iteration $B = "diag"(A)^(-1)$.

+ $1$D Poisson equation $-u^(prime prime) = 0, x in (0, 1)$ with $u(0) = u(1) = 0$. 
  $
    integral_0^1 phi_i^(prime)(x) phi_j^(prime)(x) dif x = 1/h mat(delim: "[",
      2, -1, , , ,;
      -1, 2, -1, , ,;
      , dots.down, dots.down, dots.down, ;
      , , -1, 2, -1;
      , , , -1, 2) =: 1/h A
  $
  where $phi_i (x) (i = 1,2,dots,n)$ are hat functions and $h = 1 / (n+1)$. The eigenpairs $(lambda_k, v_k)$ of $A$ are 
  $
    lambda_k (A) = 2(1 - cos(h k pi)), v_(k,j)(A) = sqrt(2 h)sin(j k h pi).
  $
  Note that $B = 1/2 I$, 
  $
  lambda_k (E) = lambda_k (I - B A) = cos(h k pi) = cos((k pi)/(n+1)).
  $
+ Numerical experiments.
  ```julia
  function f()
  end
  ```