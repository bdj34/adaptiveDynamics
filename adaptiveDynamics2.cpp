/*
    adaptiveDynamics.cpp
    Author: Gregory Kimmel
    Date: 06/12/2019

    This file calculates the evolution of neighborhood size and production traits over generations.

    USAGE
        ./run_adaptiveDynamics <inputfile> <popsize> <genMax> 
        <pTraitFile> <nsFile>

    INPUTS
        Read from terminal:
            inputfile:  File that contains fitness related parameters.
            popsize:    Size of the population.
            genMax:     The number of generations to run.
            pTraitFile: File of doubles (length: popsize*genMax) which contains
                            the evolution of production trait over generations.
            nsFile:     File of doubles (length: popsize*genMax) which contains
                            the evolution of neighborhood size over generations.
        Read from input file:
            nMin:           Minimum neighborhood size.
            nMax:           Maximum neighborhood size.
            sigma:          Freq. indep. benefit parameter
            beta:           Freq. dep. benefit parameter
            alpha:          Growth rate
            kappa:          Cost at full production P = 1
            mu:             Shape parameter of cost function
            NSsigma:        Std. dev. of neighborhood size if mutation occurs
            Psigma:         Std. dev. of production trait if mutation occurs
            mutThreshold:   Threshold for mutation to occur
            N0:             The initial neighborhood size of the population
            stdevN0:        Std. dev. of N0
            P0:             The initial production trait of the population
            stdevP0:        Std. dev. of P0
            verbosity:      Parameter which controls the output to terminal

    OUTPUTS
        Written to terminal if verbosity > 0:
            Initial population traits
        Written to terminal:
            The elapsed time
        Written to outfiles:
            productionTrait:    Array of doubles for production trait evolution
            neighborhoodSizes:  Array of doubles for neighborhood size evolution
                NOTE: both arrays are of size (genMax x popsize)

 */


#include <cstdio>
#include <cstdlib>
#include <cstdbool>
#include <string>
#include <cmath>
#include <ctime>
#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <vector>

#define ROW_COL_TO_INDEX(row, col, num_cols) (row*num_cols + col)

using namespace std;

// Details of the functions are found after main()
void initializePopulationTraits(int popsize, double nMin, double nMax,
double sigma, double beta, double alpha, double kappa, double mu,
double N0, double stdevN0, double P0, double stdevP0, double *neighborhoodSizes,
double *productionTrait, int verbosity);

double payoff(double sigma, double beta, double alpha, double kappa, double mu,
double n, int verbosity);

void runAdaptiveDynamics(int popsize, double nMin, double nMax, double sigma,
double beta, double alpha, double kappa, double mu, double NSsigma,
double Psigma, double *neighborhoodSizes, double *productionTrait, int genMax,
double mutThreshold, int verbosity);

int main(int argc, char* argv[])
{

    // Start the clock
    clock_t start = clock();

    // Check if correct usage is given
    if(argc != 6)
    {
        printf("./run_adaptiveDynamics <inputfile> <popsize> <genMax> <outfile1> <outfile2>\n");
        return -1;
    }

    // Open the input file.
    FILE* inputfile = fopen(argv[1], "r");

    // Check that the file can be opened
    if(!inputfile)
    {
        printf("Unable to open input file.\n");
        return -1;
    }

    // Initialize parameters
    double nMin, nMax, sigma, beta, alpha, kappa, mu, NSsigma, Psigma,
    mutThreshold, N0, P0, stdevN0, stdevP0;
    int verbosity;

    // Read from the input file
    if (fscanf(inputfile, "%lf", &nMin) != 1)
    {
        nMin = 1.0;
        printf("failed to read nMin. Using default value.\n");
    }
    if (fscanf(inputfile, "%lf", &nMax) != 1)
    {
        nMax = 80.0;
        printf("failed to read nMax. Using default value.\n");
    }
    if (fscanf(inputfile, "%lf", &sigma) != 1)
    {
        sigma = 2.0;
        printf("failed to read sigma. Using default value.\n");
    }
    if (fscanf(inputfile, "%lf", &beta) != 1)
    {
        beta = 5.0;
        printf("failed to read beta. Using default value.\n");
    }
    if (fscanf(inputfile, "%lf", &alpha) != 1)
    {
        alpha = 1.0;
        printf("failed to read alpha. Using default value.\n");
    }
    if (fscanf(inputfile, "%lf", &kappa) != 1)
    {
        kappa = 0.5;
        printf("failed to read kappa. Using default value.\n");
    }
    if (fscanf(inputfile, "%lf", &mu) != 1)
    {
        mu = 2.0;
        printf("failed to read mu. Using default value.\n");
    }
    if (fscanf(inputfile, "%lf", &NSsigma) != 1)
    {
        NSsigma = 0.2;
        printf("failed to read NSsigma. Using default value.\n");
    }
    if (fscanf(inputfile, "%lf", &Psigma) != 1)
    {
        Psigma = 0.005;
        printf("failed to read Psigma. Using default value.\n");
    }
    if (fscanf(inputfile, "%lf", &mutThreshold) != 1)
    {
        mutThreshold = 0.01;
        printf("failed to read mutThreshold. Using default value.\n");
    }
    if (fscanf(inputfile, "%lf", &N0) != 1)
    {
        N0 = 30.0;
        printf("failed to read N0. Using default value.\n");
    }
    if (fscanf(inputfile, "%lf", &stdevN0) != 1)
    {
        stdevN0 = 0.05;
        printf("failed to read stdevN0. Using default value.\n");
    }
    if (fscanf(inputfile, "%lf", &P0) != 1)
    {
        P0 = 0.0;
        printf("failed to read P0. Using default value.\n");
    }
    if (fscanf(inputfile, "%lf", &stdevP0) != 1)
    {
        stdevP0 = 0.05;
        printf("failed to read stdevP0. Using default value.\n");
    }
    if (fscanf(inputfile, "%d", &verbosity) != 1)
    {
        verbosity = 0;
        printf("failed to read verbosity. Using default value.\n");
    }

    // Close the input file.
    fclose(inputfile);

    // Population size
    int popsize = atoi(argv[2]);

    // Number of generations
    int genMax = atoi(argv[3]);

    // Initialize the array to be filled
    double *neighborhoodSizes = new double [genMax*popsize];
    double *productionTrait = new double [genMax*popsize];

    // Initialize the population traits
    initializePopulationTraits(popsize,nMin, nMax, sigma, beta, alpha, kappa,
        mu, N0, stdevN0, P0, stdevP0, neighborhoodSizes, productionTrait,verbosity);

    // run Adaptive dynamics
    runAdaptiveDynamics(popsize, nMin, nMax, sigma, beta, alpha, kappa,
    mu, NSsigma, Psigma, neighborhoodSizes, productionTrait, genMax, mutThreshold, verbosity);

    // Open the outputfiles.
    FILE* outfile1 = fopen(argv[4], "w");
    FILE* outfile2 = fopen(argv[5], "w");

    // Write traits to test file
    fwrite(productionTrait,sizeof(double),popsize*genMax,outfile1);
    fwrite(neighborhoodSizes,sizeof(double),popsize*genMax,outfile2);
    
    // Close the output file
    fclose(outfile1);
    fclose(outfile2);
    
    // Print the time and return.
    printf("Time elapsed: %g seconds\n", (float)(clock()-start)/CLOCKS_PER_SEC);

    // Free the memory
    delete[] neighborhoodSizes;
    delete[] productionTrait;

    return 0;
}

/*

    initializePopulationTraits

    This function initializes the individual traits of production and
    neighborhood size.

    INPUTS
        popsize - The size of the population
        nMin - the minimum neighborhood size (probably 1.0)
        nMax - the maximum neighborhood size
        sigma - The freq-independent impact of the public good
        beta - The freq-dependent impact of the public good
        alpha - The growth rate
        kappa - The cost at maximum production
        mu - Shape parameter of the cost function
        neighborhoodSizes - Array that contains an individuals neighborhood size
        productionTrait - Array that contains an individuals production trait
        verbosity - Determines the amount of output to the terminal

    OUTPUTS
        neighborhoodSizes and productionTrait initialized.
*/
void initializePopulationTraits(int popsize, double nMin, double nMax,
double sigma, double beta, double alpha, double kappa, double mu,
double N0, double stdevN0, double P0, double stdevP0, double *neighborhoodSizes,
double *productionTrait, int verbosity)
{

    // Generate the random seed
    random_device rd;
    mt19937 rng(rd());

    // Generate the distributions
    normal_distribution<double> NS_distribution(N0,N0*stdevN0);
    normal_distribution<double> prod_distribution(P0,P0*stdevP0);

    // Print information to terminal
    if (verbosity > 0)
        printf("Initial population traits:\n");

    // Initialize the population traits with a normal distribution using
    // values given by the user
    for (int i = 0; i < popsize; i++)
    {
        do
        {
            neighborhoodSizes[i] = NS_distribution(rng);
        } while (neighborhoodSizes[i]<nMin);

        do
        {
            productionTrait[i] = prod_distribution(rng);
        } while (productionTrait[i]<0 || productionTrait[i]>1);
        
        if(verbosity > 0)
            printf("%g\t%g\n",neighborhoodSizes[i],productionTrait[i]);
    }

}

/*

    payoff

    This function calculates the payoff an individual has at a particular
    production value and neighborhood size

    INPUTS
        sigma - The freq-independent impact of the public good
        beta - The freq-dependent impact of the public good
        alpha - The growth rate
        kappa - The cost at maximum production
        mu - Shape parameter of the cost function
        n - Individual's neighborhood size
        p - Individual's production trait
        meanProductionTrait - Mean production of the public good in pop.
        verbosity - Determines the amount of output to the terminal

    OUTPUTS
        payoff = benefit - cost
*/
double payoff(double sigma, double beta, double alpha, double kappa, double mu, double n, double p, double meanProductionTrait, int verbosity)
{

    // The expected amount of public good perceived by the individual
    double expectedPublicGood = (p + (n-1)*meanProductionTrait)/n;

    // Benefit function
    double benefit = alpha*(1 + exp(sigma))/
    (1+exp(sigma-beta*expectedPublicGood));

    // Cost function
    double cost = kappa*pow(tanh(p/(1.0-p)),mu);

    // Payoff function
    return benefit - cost;

}


/*

    runAdaptiveDynamics

    This function runs adaptive dynamics of popsize individuals through genMax
    generations.

    INPUTS
        popsize -   The size of the population
        nMin -      The minimum neighborhood size (probably 1.0)
        nMax -      The maximum neighborhood size
        sigma -     The freq-independent impact of the public good
        beta -      The freq-dependent impact of the public good
        alpha -     The growth rate
        kappa -     The cost at maximum production
        mu -        Shape parameter of the cost function
        NSsigma -   The std dev of the neighborhood size when mutated
        Psigma -    The std dev of the production trait when mutated
        neighborhoodSizes - Array that contains an individuals neighborhood size
        productionTrait - Array that contains an individuals production trait
        genMax -    The number of generations to run the simulation
        mutThreshold -  The probability of mutation
        verbosity - Determines the amount of output to the terminal

    OUTPUTS
        neighborhoodSizes and productionTrait are filled.


*/
void runAdaptiveDynamics(int popsize, double nMin, double nMax, double sigma,
double beta, double alpha, double kappa, double mu, double NSsigma,
double Psigma, double *neighborhoodSizes, double *productionTrait, int genMax,
double mutThreshold, int verbosity)
{
    // Random seed
    random_device rd;
    mt19937 rng(rd());

    // Initialize the distribution to select a random individual
    // and the distribution to decide which event occurs
    uniform_int_distribution<int> uniInt(0,popsize-1);
    uniform_real_distribution<double> randReal(0.0,1.0);
    normal_distribution<double> perturbNS(0.0,NSsigma);
    normal_distribution<double> perturbProduction(0.0,Psigma);

    // Initialize random individual to play game against and meanProduction of
    // public good.
    int randIndividual, individual, opponent, offspring, i, k, h, indexIndividual, indexOpponent;
    double meanProductionIndividual, meanProductionOpponent, payoffDiff, w, roll, changeNS, changeProduction;
    double maxPayoff, minPayoff, sumIndividual, sumOpponent;

    //Make a 0 to popsize-1 int matrix increasing by 1
    vector<int> indices;
    for (h=0; h<popsize; h++)
    {
        indices.push_back(h);
    }

    // Loop through generations
    for (int nGen = 0; nGen < genMax-1; nGen++)
    {
        //printf("%d\n", nGen);
        // The maximum and minimum payoff in that generation
        maxPayoff = *max_element(
            productionTrait + ROW_COL_TO_INDEX(nGen,0,popsize),
        productionTrait + ROW_COL_TO_INDEX(nGen,popsize,popsize));
        minPayoff = *min_element(
            productionTrait + ROW_COL_TO_INDEX(nGen,0,popsize),
        productionTrait + ROW_COL_TO_INDEX(nGen,popsize,popsize));

        // Loop through population
        for (i = 0; i < popsize; i++)
        {
            // Randomly select an individual to compete with
            do
            {
                randIndividual = uniInt(rng);
            } while (randIndividual == i);

            // Get the index of the individual and their opponent
            individual = ROW_COL_TO_INDEX(nGen,i,popsize);
            opponent = ROW_COL_TO_INDEX(nGen,randIndividual,popsize);
            
            //Shuffle the deck...random index so random numbers from 0 to popsize-1
            random_device rd;
            mt19937 g(rd());
            shuffle(begin(indices), end(indices),g);
            
            
            //Randomly select neighbors making sure the individual isn't the neighbor
            int individualNS= (int) neighborhoodSizes[individual];
            int *individualNeighborIndex = new int [individualNS];
            int counter=0;
            for (k=0; k < popsize; k++)
            {   
                if (indices[k]==i)
                    continue;
                else
                {
                    individualNeighborIndex[counter]=indices[k];
                    counter++;
                }
                if (counter>individualNS-1)
                {
                    break;
                }
            }    
            //Get the sum of the neighbors of the individual
            int *individualNeighbors = new int [individualNS];
            for (k=0; k<individualNS; k++)
            {
                sumIndividual+=productionTrait[ROW_COL_TO_INDEX(nGen,individualNeighborIndex[k],popsize)];
            }

            //Shuffle again
            shuffle(begin(indices), end(indices),g);
            
            //Randomly select opponent neighbors making sure the opponent isn't the neighbor
            int opponentNS= (int) neighborhoodSizes[opponent];
            int *opponentNeighborIndex = new int [opponentNS];
            counter=0;
            for (k=0; k < popsize; k++)
            {   
                if (indices[k]==randIndividual)
                    continue;
                else
                {
                    opponentNeighborIndex[counter]=indices[k];
                    counter++;
                }
                if (counter>opponentNS-1)
                {
                    break;
                }
            }    

            //Get the sum of the neighbors of the opponent
            int *opponentNeighbors = new int [opponentNS];
            for (k=0; k<opponentNS-1; k++)
            {
                sumOpponent+=productionTrait[ROW_COL_TO_INDEX(nGen,opponentNeighborIndex[k],popsize)];
            }

            // Mean production of the neighborhood excluding the focal at nGen
            meanProductionIndividual = sumIndividual/(individualNS-1);
            meanProductionOpponent = sumOpponent/(opponentNS-1);

            // Once we select an individual, we look at the payoff difference
            // payoff(opponent) - payoff(individual)
            payoffDiff = payoff(sigma, beta, alpha, kappa, mu,
            neighborhoodSizes[opponent], productionTrait[opponent],
            meanProductionOpponent, verbosity) -
            payoff(sigma, beta, alpha, kappa, mu, neighborhoodSizes[individual],
            productionTrait[individual], meanProductionIndividual, verbosity); 

            // Normalizing the payoff difference. If the max payoff difference
            // is 0, we assume a 50-50 change to replace either individual.
            w = (maxPayoff-minPayoff>__DBL_EPSILON__)?
            payoffDiff/(maxPayoff-minPayoff):0.5;

            // We now roll to see whether the opponent will replace the 
            // individual in the game.
            roll = randReal(rng);

            // Index for the offspring
            // (NOTE: the () around first argument!)
            offspring = ROW_COL_TO_INDEX((nGen+1),i,popsize);

            // The parent is the individual
            if (w < roll)
            { 
                productionTrait[offspring] = productionTrait[individual];
                neighborhoodSizes[offspring] = neighborhoodSizes[individual];
            }
            else
            {
                productionTrait[offspring] = productionTrait[opponent];
                neighborhoodSizes[offspring] = neighborhoodSizes[opponent];
            }

            // Now we check for mutations: first up is neighborhood size
            roll = randReal(rng);

            // Mutation in neighborhood size
            if (roll < mutThreshold)
            {
                do
                {
                    changeNS = perturbNS(rng);
                } while (changeNS + neighborhoodSizes[offspring] < nMin);

                neighborhoodSizes[offspring] += changeNS;

            }
            
            // Check mutation for production trait
            roll = randReal(rng);

            // Mutation in neighborhood size
            if (roll < mutThreshold)
            {
                do
                {
                    changeProduction = perturbProduction(rng);
                } while (changeProduction + productionTrait[offspring] < 0 ||
                changeProduction + productionTrait[offspring] > 1);

                productionTrait[offspring] += changeProduction;
            }
            delete[] individualNeighbors;
            delete[] individualNeighborIndex;
            delete[] opponentNeighbors;
            delete[] opponentNeighborIndex;
        }
    }

    

}

