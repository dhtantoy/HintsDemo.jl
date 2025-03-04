#import "@preview/touying:0.6.1": *
#import themes.university: *
#import "@preview/cetz:0.3.2"
#import "@preview/fletcher:0.5.4" as fletcher: node, edge
#import "@preview/numbly:0.1.0": numbly
#import "@preview/theorion:0.3.2": *
#import cosmos.clouds: *
#import "@preview/codly:1.2.0": *
#import "@preview/codly-languages:0.1.1": *
#import "@preview/algorithmic:0.1.0"
#import algorithmic: algorithm
#show: codly-init.with()
#show: show-theorion

#codly(
  languages: (
    julia: (name: "Julia", color: rgb("#9558B2")),
  )
)

#show link: set text(fill: red)

// cetz and fletcher bindings for touying
#let cetz-canvas = touying-reducer.with(reduce: cetz.canvas, cover: cetz.draw.hide.with(bounds: true))
#let fletcher-diagram = touying-reducer.with(reduce: fletcher.diagram, cover: fletcher.hide)

#show: university-theme.with(
  aspect-ratio: "16-9",
  // align: horizon,
  // config-common(handout: true),
  config-common(frozen-counters: (theorem-counter,)),  // freeze theorem counter for animation
  config-info(
    title: [A Brief Introduction to HINTS],
    // subtitle: [Hybrid Iterative Numerical Transferable Solver],
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

= Overview
== What is HINTS
HINTS@hints is a hybrid, iterative, numerical, and transferable
solver for differential equations which combines standard relaxation methods and the Deep Operator Network (DeepONet).
#image("images/hints.png", width: 100%)

= Basic Principles
== Smoothing Principles
+ A basic residual-correction method to solve $A u = f$ for a given initial guess $u_0$.
  + residual $r = f - A u_k$.
  + correction $e = B r$ with $B approx A^(-1)$.
  + $u_(k+1) = u_k + e$.
+ Amplification matrix.
  $ e_(k+1) &:= u - u_(k+1) = (I - B A) (u-u_k) = (I - B A)e_k
  $
  #theorem()[
    The residual-correction shceme converges iff 
    $
      rho(I - B A) < 1.
    $
  ]

+ Jacobi iteration $B = "diag"(A)^(-1)$.

+ $-u^(prime prime) = 0, x in (0, 1)$ with $u(0) = u(1) = 0$. 
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
  lambda_k (I - B A) = cos(h k pi) = cos((k pi)/(n+1)).
  $
  #image("images/modes.svg", height: 100%)
  #image("images/modes_residual.svg", height: 100%)

== Frequency Principle
Frequency principle@frequency_principle indicates that deep neural networks always approximate the low-frequency part of the target function first, and then approach the high-frequency part. Here is an example. #footnote[see https://github.com/dhtantoy/HintsDemo.jl.git for more details.]
  $
    -u^(prime prime) &= 1/10 sin(pi x) + sin(1000 pi x), x in (0, 1), \
    u(0) &= u(1) = 0.
  $
  #align(center)[#image("images/NN.png", height: 100%)]


= Algorithms

== Residual Correction
#algorithm({
  import algorithmic: *
  Function("iteration", args: ("A", [$bold(b)$], "epochs", $epsilon$), {
    // Cmt[Initialize the search range]
    Assign[$bold(v)$][$bold(0)$]
    Assign[$i$][$0$]
    Assign[$bold(r)$][$bold(b) - A bold(v)$]
    State[]
    While(cond: [$norm(bold(r)) <= epsilon$ and $i <= "epochs"$], {
      State([solve $A bold(delta v) = bold(r)$])
      Assign[$bold(v)$][$bold(v) + bold(delta v)$]
      Assign[$i$][$i + 1$]
    })

    Return[$bold(v)$]
  })
})

== Residual Equations
#algorithm({
  import algorithmic: *
  Function("deeponet_jacobi", args: ("A", [$bold(r)$], $p$), {
    // Cmt[Initialize the search range]
    Assign[$bold(delta v)$][$"DeepONet"(bold(r))$]
    Assign[$bold(delta v^0)$][$bold(delta v)$]
    Assign[$s$][$0$]
    State[]
    While(cond: $s < p$, {
      State([solve $A bold(delta v) = bold(r)$])
      Assign[$bold(delta v)_i^(s+1)$][$1/a_(i i)(bold(r) - sum_(j=1)^n a_(i j) bold(delta v)_j^s) $]
      Assign[$s$][$s + 1$]
    })

    Return[$bold(delta v)^p$]
  })
})

= Numerical Experiments

== 1D Poisson Equation
$
  -u^(prime prime) &= f, x in (0, 1), \
  u^prime (0) + u(0) &= c_0, \
  u^prime (1) + u(1) &= c_1.
$
Here $f = pi^2 sin(pi x) + 25pi^2 sin(5pi x)$ and $c_0 = c_1 = 0$. The exact solution is $u = sin(pi x) + sin(5pi x)$.
#image("images/possion_1.png", height: 100%)

== 2D Helmholtz Equation
$
  -Delta u - k^2 u &= f, & "in" Omega, \
  nabla u dot bold(n) + u &= g, & "on" partial Omega,
$
where $Omega = (0, 1)^2, f(x, y) = -x^2 - y^2 - exp(-y), g(x, y) = sin(x)sin(y)$, and
$
  k(x, y) = cases(
    10 "if" (x, y) in [0,0.5]^2 union [0.5, 1]^2 ",", \
    20 "if" (x, y) in [0, 0.5) times (0.5, 1] union (0.5, 1] times [0, 0.5).
  )
$
#align(center)[
  #box(image("images/helmholtz.svg", height: 100%), clip: true, inset: (right: -12.8in))
]


== 2D Stokes Equations
$
  -nabla dot (mu nabla bold(u)) + nabla p &= bold(f), & "in" Omega, \
  nabla dot bold(u) &= 0, & "in" Omega, \
  (nabla bold(u) - p I) dot bold(n) + bold(u) &= bold(0), & "on" partial Omega,
$
where $Omega = (0, 1)^2, mu = 1$ and 
$
  bold(f)(x, y) = cases(
    mat(1, 0)^top "if" (x, y) in [0.3, 0.4] times [0.2, 0.3] ",", \
    mat(-1, 0)^top "if" (x, y) in [0.3, 0.4] times [0.7, 0.8] ",", \
    mat(0, 0)^top "otherwise".
  )
$
#align(center)[
  #box(image("images/stokes.png", width: 100%), clip: true, inset: (top: -4.0in, right: -5in))
]
#align(center)[
  #box(image("images/stokes_ex.svg", width: 100%), clip: true, inset:(right: -5in))
]

= Conclusions
== Pros and Cons
- Retains the features of operator learning, allowing the same model to solve equations with different boundary conditions.
- Effectively accelerates the convergence speed of stationary iteration, especially for Jacobi iteration.
- Can use models trained on coarse grids to solve discrete problems on fine grids.
- Can use models trained with low-order finite elements to solve problems discretized with high-order finite elements.
- The algorithm only involves matrix-vector multiplication.


- Training of DeepONet is costly.
- Difficult to approximate high-frequency target functions.
