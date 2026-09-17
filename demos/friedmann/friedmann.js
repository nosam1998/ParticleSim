/*
 * Friedmann integrators, in the browser and in a test harness.
 *
 * This file is the physics behind demos/friedmann/index.html and nothing
 * else: no DOM, no canvas, no globals beyond the one export. That is on
 * purpose. The demo's claim is that it reproduces the Python background
 * solvers rather than illustrating them, and a claim like that has to be
 * checkable, so the same file is evaluated by
 * tests/unit/test_cosmo_views.py and its numbers are compared with
 * particlesim.cosmo.dynamics.evolve and particlesim.cosmo.background.
 *
 * Two solvers, matching the two Python modules:
 *
 *   theoryRun  <-> particlesim.cosmo.dynamics.evolve
 *     Geometric units, G = c = 1, densities in Planck units. Integrates
 *     H' = -(3/2)(1+w) rho f'(rho) with the plugin's own H^2 = f(rho),
 *     second order so that a bounce -- where f = 0 and sqrt(f) has a
 *     branch point -- is crossed rather than stopped at. The constraint
 *     H^2 = f(rho) is monitored, not imposed, and the drift is reported.
 *
 *   lcdm       <-> particlesim.cosmo.background.Cosmology
 *     Observational units: H0 in km/s/Mpc, distances in Mpc, times in Gyr.
 *     The same two substitutions carry the accuracy -- distances over
 *     ln(1+z) and ages over s with a = s^2 -- because in redshift the
 *     distance integrand spans decades and over a the age integrand has an
 *     infinite second derivative at zero.
 *
 * Where the Python uses Gauss-Legendre this uses Simpson's rule on the
 * same substituted integrands. Fixed-order Gauss-Legendre needs its nodes,
 * which means a Newton solve for the roots of a Legendre polynomial; on
 * integrands this smooth Simpson with a few thousand intervals gets to
 * 1e-12 of the same answer for a few lines of code, and the page reports
 * its own convergence by halving the interval count.
 */
(function (root) {
  "use strict";

  /* 8 pi / 3, the coefficient of the general-relativistic Friedmann equation. */
  var FRIEDMANN = 8 * Math.PI / 3;
  /* Critical density in Planck units, from the loop quantum gravity area gap. */
  var RHO_CRITICAL_PLANCK = 0.41;
  /* Speed of light in km/s, and the megaparsec in km. */
  var LIGHT_KM_S = 299792.458;
  var MPC_KM = 3.0856775814913673e19;
  /* Seconds in a gigayear on the Julian year: 365.25 * 86400 * 1e9 exactly. */
  var GYR_S = 365.25 * 86400.0 * 1e9;

  var THEORIES = {
    "gr": {
      label: "General relativity",
      hubbleSquared: function (rho) { return FRIEDMANN * rho; },
      note: "H² = (8π/3)ρ. No bounce: a collapse reaches a = 0."
    },
    "lqg.lqc": {
      label: "Effective loop quantum cosmology",
      hubbleSquared: function (rho) {
        return FRIEDMANN * rho * (1 - rho / RHO_CRITICAL_PLANCK);
      },
      note: "H² = (8π/3)ρ(1 − ρ/ρ_c), ρ_c = 0.41. The big bang becomes a bounce."
    }
  };

  /* Density at which H^2 passes through zero, or null if it never does.
   *
   * Bisection on the theory's own H^2(rho), like the Python's brentq: the
   * answer is then exact to the solver's tolerance and does not depend on
   * any step size taken during an evolution.
   */
  function bounceDensity(theory, lower, upper, tolerance) {
    lower = lower === undefined ? 1e-12 : lower;
    upper = upper === undefined ? 1e6 : upper;
    tolerance = tolerance === undefined ? 1e-14 : tolerance;
    var f = theory.hubbleSquared;
    if (f(lower) <= 0) { throw new Error("H^2 is already non-positive at the lower bracket"); }
    if (f(upper) > 0) { return null; }
    var low = lower, high = upper;
    for (var i = 0; i < 300 && high - low > tolerance * Math.max(1, high); i += 1) {
      var mid = 0.5 * (low + high);
      if (f(mid) > 0) { low = mid; } else { high = mid; }
    }
    return 0.5 * (low + high);
  }

  /* Integrate a theory's FLRW dynamics through whatever it does.
   *
   * Classical fourth-order Runge-Kutta on [a, H] with a fixed step. The
   * Python uses an adaptive eighth-order method at rtol 1e-11; the two
   * agree on the bounce density to better than a part in a million, which
   * is what the page reports and what the test asserts.
   */
  function theoryRun(options) {
    var opts = options || {};
    var theory = opts.theory || THEORIES.gr;
    var w = opts.equationOfState === undefined ? 0 : opts.equationOfState;
    var density = opts.density === undefined ? 1e-6 : opts.density;
    var scaleFactor = opts.scaleFactor === undefined ? 1 : opts.scaleFactor;
    var contracting = opts.contracting === undefined ? true : opts.contracting;
    var steps = opts.steps === undefined ? 400000 : opts.steps;
    var samples = opts.samples === undefined ? 800 : opts.samples;
    var f = theory.hubbleSquared;

    var exponent = -3 * (1 + w);
    var coefficient = density * Math.pow(scaleFactor, -exponent);
    function densityOf(a) { return coefficient * Math.pow(a, exponent); }
    function derivativeOfF(rho) {
      var step = 1e-6 * Math.max(Math.abs(rho), 1e-30);
      return (f(rho + step) - f(rho - step)) / (2 * step);
    }

    var initial = f(density);
    if (initial < 0) { throw new Error("the starting density is past this theory's bounce"); }
    var hubble0 = Math.sqrt(initial) * (contracting ? -1 : 1);
    var duration = opts.duration === undefined ? 2.5 / Math.abs(hubble0) : opts.duration;

    function rhs(state) {
      var rho = densityOf(state[0]);
      return [state[0] * state[1], -1.5 * (1 + w) * rho * derivativeOfF(rho)];
    }

    /* a = 0 is reached in finite time under general relativity, and no
     * fixed step can follow it: the solution is singular there, and the
     * constraint residual blows up with it. The floor here is a thousandth
     * of the starting scale factor rather than the Python solver's 1e-8,
     * because that is where a fixed-step method stops being able to say
     * anything -- and the picture is the same either way, since the point
     * of the run is that general relativity crunches where loop quantum
     * cosmology turns around. */
    var floor = 1e-3 * scaleFactor;
    var h = duration / steps;
    var every = Math.max(1, Math.floor(steps / Math.max(1, samples)));
    var state = [scaleFactor, hubble0];
    var time = [0], scale = [state[0]], hubble = [state[1]];
    var bounce = null, crunched = false, drift = 0, scaleOfDrift = 0;

    for (var n = 0; n < steps; n += 1) {
      var previous = [state[0], state[1]];
      var k1 = rhs(state);
      var k2 = rhs([state[0] + 0.5 * h * k1[0], state[1] + 0.5 * h * k1[1]]);
      var k3 = rhs([state[0] + 0.5 * h * k2[0], state[1] + 0.5 * h * k2[1]]);
      var k4 = rhs([state[0] + h * k3[0], state[1] + h * k3[1]]);
      state = [
        state[0] + (h / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
        state[1] + (h / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
      ];
      var t = (n + 1) * h;

      /* A collapse can step through a = 0, and then a^(-3(1+w)) is
       * infinite or not a number and every later sample is NaN. Catching
       * it here means a crunch ends the run where it happened instead of
       * poisoning the arrays behind it. */
      if (!isFinite(state[0]) || !isFinite(state[1]) || state[0] <= floor) {
        crunched = true;
        break;
      }

      if (bounce === null && previous[1] < 0 && state[1] >= 0) {
        /* rho is stationary at the turning point -- rho' = -3H(1+w)rho and
         * H = 0 there -- so interpolating the crossing linearly costs only
         * O(h^2) in the density, which is why the peak value can be read
         * off a fixed-step run at all. */
        var fraction = -previous[1] / (state[1] - previous[1]);
        var at = previous[0] + fraction * (state[0] - previous[0]);
        bounce = { time: t - h + fraction * h, scaleFactor: at, density: densityOf(at) };
      }
      var expected = f(densityOf(state[0]));
      var residual = Math.abs(state[1] * state[1] - expected);
      drift = Math.max(drift, residual);
      scaleOfDrift = Math.max(scaleOfDrift, Math.abs(expected));
      /* The other way a collapse ends: not at the floor but where a fixed
       * step stops being able to follow the solution, which the monitored
       * constraint says before the scale factor does. The residual is
       * measured against the largest H^2 the run has seen rather than the
       * current one, because at a bounce the current one is zero and every
       * relative test would fire there. */
      if (residual > 1e-3 * scaleOfDrift) {
        crunched = true;
        time.push(t); scale.push(state[0]); hubble.push(state[1]);
        break;
      }
      /* The integration is fine-grained and the output is not: recording
       * every step would hand the page a ten-megabyte array to draw eight
       * hundred pixels with. */
      if ((n + 1) % every === 0 || n === steps - 1) {
        time.push(t); scale.push(state[0]); hubble.push(state[1]);
      }
    }

    return {
      time: time, scaleFactor: scale, hubble: hubble,
      bounce: bounce, crunched: crunched,
      bounced: bounce !== null,
      equationOfState: w,
      constraintDrift: scaleOfDrift > 0 ? drift / scaleOfDrift : 0,
      duration: duration
    };
  }

  /* Composite Simpson's rule for integral_0^upper f, with n intervals. */
  function simpson(f, upper, n) {
    n = n === undefined ? 2048 : 2 * Math.ceil(n / 2);
    if (upper === 0) { return 0; }
    var h = upper / n;
    var total = f(0) + f(upper);
    for (var i = 1; i < n; i += 1) {
      total += (i % 2 === 0 ? 2 : 4) * f(i * h);
    }
    return total * h / 3;
  }

  /* A flat-or-curved FLRW background in observational units. */
  function lcdm(options) {
    var opts = options || {};
    var h0 = opts.h0 === undefined ? 67.36 : opts.h0;
    var omegaR = opts.omegaR === undefined ? 0 : opts.omegaR;
    var omegaM = opts.omegaM === undefined ? 0.3153 : opts.omegaM;
    var omegaL = opts.omegaL === undefined ? 0.6847 : opts.omegaL;
    var order = opts.order === undefined ? 2048 : opts.order;
    var omegaK = 1 - omegaR - omegaM - omegaL;
    var hubbleDistance = LIGHT_KM_S / h0;
    var hubbleTime = MPC_KM / (h0 * GYR_S);

    function rateSquaredOfA(a) {
      return omegaR / Math.pow(a, 4) + omegaM / Math.pow(a, 3) + omegaK / (a * a) + omegaL;
    }
    function expansionRateSquared(z) { return rateSquaredOfA(1 / (1 + z)); }
    function hubble(z) { return h0 * Math.sqrt(expansionRateSquared(z)); }

    function comovingDistance(z, n) {
      /* Over u = ln(1+z): in redshift the integrand spans decades and a
       * fixed rule spreads its nodes evenly across all of it. */
      var upper = Math.log1p(z);
      return hubbleDistance * simpson(function (u) {
        return Math.exp(u) / Math.sqrt(expansionRateSquared(Math.expm1(u)));
      }, upper, n === undefined ? order : n);
    }

    function age(z, n) {
      /* Over s with a = s^2: over a the integrand goes as a^(1/2) near
       * zero, whose second derivative is infinite there. */
      var end = 1 / (1 + (z === undefined ? 0 : z));
      return hubbleTime * simpson(function (s) {
        if (s <= 0) { return 0; }
        return 2 / (s * Math.sqrt(rateSquaredOfA(s * s)));
      }, Math.sqrt(end), n === undefined ? order : n);
    }

    function luminosityDistance(z) {
      var comoving = comovingDistance(z);
      if (Math.abs(omegaK) < 1e-12) { return (1 + z) * comoving; }
      var root = Math.sqrt(Math.abs(omegaK)) * comoving / hubbleDistance;
      var transverse = omegaK > 0
        ? hubbleDistance * Math.sinh(root) / Math.sqrt(omegaK)
        : hubbleDistance * Math.sin(root) / Math.sqrt(-omegaK);
      return (1 + z) * transverse;
    }

    /* a(t) from the same quadrature the age uses: t(a) is the age at that
     * scale factor, so the expansion history costs nothing extra. */
    function history(points) {
      points = points === undefined ? 160 : points;
      var out = { time: [], scaleFactor: [] };
      for (var i = 1; i <= points; i += 1) {
        var a = i / points;
        out.scaleFactor.push(a);
        out.time.push(age(1 / a - 1));
      }
      return out;
    }

    return {
      h0: h0, omegaR: omegaR, omegaM: omegaM, omegaL: omegaL, omegaK: omegaK,
      hubbleDistance: hubbleDistance, hubbleTime: hubbleTime,
      hubble: hubble, comovingDistance: comovingDistance, age: age,
      luminosityDistance: luminosityDistance, history: history
    };
  }

  root.Friedmann = {
    FRIEDMANN: FRIEDMANN,
    RHO_CRITICAL_PLANCK: RHO_CRITICAL_PLANCK,
    LIGHT_KM_S: LIGHT_KM_S,
    GYR_S: GYR_S,
    THEORIES: THEORIES,
    bounceDensity: bounceDensity,
    theoryRun: theoryRun,
    simpson: simpson,
    lcdm: lcdm
  };
})(typeof globalThis !== "undefined" ? globalThis : this);
