# euler1d_electrostatic

1D Euler equation model for flow across an ion thruster from emitter to collector.

## How to Use

`model.py` contains functions for both incompressible modeling and compressible modeling. Both rely on analytical solutions for electric field, `E`, and current density, `rho_c`. A numerical solver for the current density, `j` is provided in `solve_j()` and used as inputs for the electric field and charge density.

The incompressible model, `model_incomp()` solves for pressure and electric field numerically using an Euler forward integration method. The analytical solutions as presented in "Ion Drag Pumps" by Otmar M. Stuetzer (1959) are coded as point of comparison.

The compressible model, `model()` solves for pressure, velocity, and density using the Euler equations with electrostatic body forces proportional to the electric field and charge density. From an initial guess for these variables at all points in the discretized space, the solver iterates using Newton's method and central differences estimation for derivatives. The equations implemented are detailed in the PowerPoint. The jacobian matrix is computed numerically using small perturbations in each variable at each point in space.

Constants and boundary conditions are set at the top of the file.
