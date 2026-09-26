//------------------------------------------------------------------------------
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef double (*LossFunction)(const double *x, int n);


/*
 * Convert normalized parameter z[i] in [0,1]
 * to the physical parameter x[i].
 */
static double denormalize(
    double z,
    double lower,
    double upper)
{
    return lower + z * (upper - lower);
}


/*
 * Evaluate the black-box loss using normalized parameters.
 *
 * z      : normalized parameters, each in [0,1]
 * x      : temporary physical parameters
 * lower  : physical lower bounds
 * upper  : physical upper bounds
 * n      : number of parameters
 */
static double evaluate_loss(
    LossFunction loss,
    const double *z,
    double *x,
    const double *lower,
    const double *upper,
    int n)
{
    for (int i = 0; i < n; ++i) {
        x[i] = denormalize(z[i], lower[i], upper[i]);
    }

    return loss(x, n);
}


/*
 * Gradient descent in normalized coordinates.
 *
 * The optimizer operates on z[i] in [0,1].
 * The user's loss function still receives the actual
 * physical parameters x[i].
 */
double gradient_descent(
    LossFunction loss,
    double *x,
    int n,
    const double *lower,
    const double *upper,
    double learning_rate,
    double finite_diff,
    int max_iter,
    double tolerance)
{
    double *z = malloc(n * sizeof(double));
    double *gradient = malloc(n * sizeof(double));
    double *xtmp = malloc(n * sizeof(double));

    if (!z || !gradient || !xtmp) {
        fprintf(stderr, "Memory allocation failed\n");
        free(z);
        free(gradient);
        free(xtmp);
        return NAN;
    }

    /*
     * Convert initial physical parameters to normalized coordinates.
     */
    for (int i = 0; i < n; ++i) {
        if (upper[i] <= lower[i]) {
            fprintf(stderr, "Invalid range for parameter %d\n", i);

            free(z);
            free(gradient);
            free(xtmp);

            return NAN;
        }

        z[i] = (x[i] - lower[i]) /
               (upper[i] - lower[i]);

        /*
         * Ensure the starting point is inside the range.
         */
        if (z[i] < 0.0)
            z[i] = 0.0;

        if (z[i] > 1.0)
            z[i] = 1.0;
    }


    for (int iter = 0; iter < max_iter; ++iter) {

        /*
         * Calculate gradient with respect to NORMALIZED
         * parameters z[i].
         */
        for (int i = 0; i < n; ++i) {

            double original = z[i];

            /*
             * Use central difference when possible.
             */
            if (original - finite_diff >= 0.0 &&
                original + finite_diff <= 1.0) {

                z[i] = original + finite_diff;
                double f_plus = evaluate_loss(
                    loss, z, xtmp,
                    lower, upper, n);

                z[i] = original - finite_diff;
                double f_minus = evaluate_loss(
                    loss, z, xtmp,
                    lower, upper, n);

                gradient[i] =
                    (f_plus - f_minus) /
                    (2.0 * finite_diff);
            }

            /*
             * Forward difference at lower boundary.
             */
            else if (original + finite_diff <= 1.0) {

                z[i] = original + finite_diff;
                double f_plus = evaluate_loss(
                    loss, z, xtmp,
                    lower, upper, n);

                z[i] = original;
                double f0 = evaluate_loss(
                    loss, z, xtmp,
                    lower, upper, n);

                gradient[i] =
                    (f_plus - f0) /
                    finite_diff;
            }

            /*
             * Backward difference at upper boundary.
             */
            else {

                z[i] = original - finite_diff;
                double f_minus = evaluate_loss(
                    loss, z, xtmp,
                    lower, upper, n);

                z[i] = original;
                double f0 = evaluate_loss(
                    loss, z, xtmp,
                    lower, upper, n);

                gradient[i] =
                    (f0 - f_minus) /
                    finite_diff;
            }

            z[i] = original;
        }


        /*
         * Calculate gradient norm.
         */
        double gradient_norm = 0.0;

        for (int i = 0; i < n; ++i)
            gradient_norm += gradient[i] * gradient[i];

        gradient_norm = sqrt(gradient_norm);


        /*
         * Current loss.
         */
        double current_loss =
            evaluate_loss(
                loss, z, xtmp,
                lower, upper, n);


        printf(
            "Iteration %4d: loss = %.12g, "
            "|gradient| = %.6g\n",
            iter,
            current_loss,
            gradient_norm
        );


        /*
         * Convergence.
         */
        if (gradient_norm < tolerance)
            break;


        /*
         * Gradient descent in normalized coordinates.
         */
        for (int i = 0; i < n; ++i) {

            z[i] -= learning_rate * gradient[i];

            /*
             * Enforce normalized bounds.
             */
            if (z[i] < 0.0)
                z[i] = 0.0;

            if (z[i] > 1.0)
                z[i] = 1.0;
        }
    }


    /*
     * Convert optimized normalized parameters
     * back to physical parameters.
     */
    for (int i = 0; i < n; ++i) {
        x[i] = denormalize(
            z[i],
            lower[i],
            upper[i]);
    }


    double final_loss =
        evaluate_loss(
            loss, z, xtmp,
            lower, upper, n);

    free(z);
    free(gradient);
    free(xtmp);

    return final_loss;
}


/*
 * ---------------------------------------------------------
 * Example black-box loss function
 * ---------------------------------------------------------
 */
double my_loss(const double *x, int n)
{
    (void)n;

    /*
     * Physical parameters have very different scales:
     *
     * x[0] ~ 0 ... 100000
     * x[1] ~ 0 ... 1
     * x[2] ~ 0 ... 0.001
     *
     * This could just as easily be a simulation or
     * experimental measurement instead of this formula.
     */

    double e0 = (x[0] - 75000.0) / 100000.0;
    double e1 = (x[1] - 0.75)     / 1.0;
    double e2 = (x[2] - 0.0004)   / 0.001;

    return e0 * e0 +
           e1 * e1 +
           e2 * e2;
}


int main(void)
{
    const int n = 3;

    /*
     * Initial physical parameters.
     */
    double x[] = {
        20000.0,
        0.20,
        0.0008
    };

    /*
     * Physical search ranges.
     */
    double lower[] = {
        0.0,
        0.0,
        0.0
    };

    double upper[] = {
        100000.0,
        1.0,
        0.001
    };


    double final_loss = gradient_descent(
        my_loss,
        x,
        n,
        lower,
        upper,

        /*
         * Learning rate in normalized coordinates.
         */
        0.1,

        /*
         * Finite-difference step in normalized coordinates.
         */
        1e-5,

        /*
         * Maximum iterations.
         */
        1000,

        /*
         * Convergence tolerance.
         */
        1e-8
    );


    printf("\nOptimized parameters:\n");

    for (int i = 0; i < n; ++i)
        printf(
            "x[%d] = %.12g "
            "(range %.12g ... %.12g)\n",
            i,
            x[i],
            lower[i],
            upper[i]
        );

    printf("Final loss = %.12g\n", final_loss);

    return 0;
}

