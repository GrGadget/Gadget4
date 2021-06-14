#pragma once

#include <gsl/gsl_integration.h>  // gsl_integration_workspace
#include <cmath>                  // sqrt, pow

namespace gadget{

class linear_growth
{
  double Omega0, OmegaLambda;
  double growthfactor_integrand(double a)
  {
    return std::pow(a / (Omega0 + (1 - Omega0 - OmegaLambda) * a + OmegaLambda * a * a * a), 1.5);
  }

 public:
  linear_growth(double in_Omega0, double in_OmegaLambda) : Omega0{in_Omega0}, OmegaLambda{in_OmegaLambda} {}

  double ratio(double astart, double aend) { return (*this)(aend) / (*this)(astart); }

  double operator()(double a)
  {
    double hubble_a    = std::sqrt(Omega0 / (a * a * a) + (1 - Omega0 - OmegaLambda) / (a * a) + OmegaLambda);
    const int worksize = 100000;
    double result, abserr;
    gsl_function F;

    gsl_integration_workspace *workspace = gsl_integration_workspace_alloc(worksize);
    F.function                           = &growthfactor_integrand;
    gsl_integration_qag(&F, 0, a, 0, 1.0e-8, worksize, GSL_INTEG_GAUSS41, workspace, &result, &abserr);
    gsl_integration_workspace_free(workspace);
    return hubble_a * result;
  }
};

}
