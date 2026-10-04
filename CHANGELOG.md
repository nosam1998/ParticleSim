# Changelog

## 0.2.2 (2026-10-04)

- Finish movie encoding without flushing a closed input pipe, preserving
  ffmpeg diagnostics when encoding fails.
- Pad odd-sized figures by at most one pixel for H.264 playback.

## 0.2.1 (2026-10-04)

- Replace the outdated foundations-only README status with the implemented
  milestone scope and a linked issue/acceptance audit.
- Document semantic versions in branch names and PR titles.

## 0.2.0 (2026-10-04)

- Add explicitly bounded axes to generated NumPy and JAX kernels, with
  second-, fourth- and sixth-order first, second and mixed derivatives.
- Preserve sixth-order accuracy in the radiative boundary's edge stencils.
- Fix generated kernels that use a field only through its derivatives.

The older 1.0.0 entry below belongs to the repository template; ParticleSim's
package version started at 0.1.0.

## 1.0.0 (2025-04-20)


### Features

* Added changelog and contributing markdown files ([fb3c7d9](https://github.com/CurtisTechSolutions/base-repo-template/commit/fb3c7d94b68b51983273c7e8583aeaa7e58651a2))
* Added docker build and publish action ([a05aed3](https://github.com/CurtisTechSolutions/base-repo-template/commit/a05aed364e610dd41d3364a9b520d6b6f12be417))
* Added GPL 3.0 License ([1ddc234](https://github.com/CurtisTechSolutions/base-repo-template/commit/1ddc234eb50ef7d5942a9c06b98d0e427d47e857))
* Added release please action ([e351f36](https://github.com/CurtisTechSolutions/base-repo-template/commit/e351f361a4f4e332143c080a4b0cffba3cfb3e32))
* Updated GH actions to correct location ([128607c](https://github.com/CurtisTechSolutions/base-repo-template/commit/128607c33a4d90f72ba677aa1a3cd3a7d3519dc1))
