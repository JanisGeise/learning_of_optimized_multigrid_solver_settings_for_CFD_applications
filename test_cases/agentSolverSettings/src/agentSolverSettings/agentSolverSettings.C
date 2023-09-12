/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2015-2016 OpenFOAM Foundation
    Copyright (C) 2015-2021 OpenCFD Ltd.
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "agentSolverSettings.H"
#include "addToRunTimeSelectionTable.H"
#include "volFields.H"
#include "ListOps.H"
#include "zeroGradientFvPatchField.H"
#include "dictionary.H"
#include "IOdictionary.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace functionObjects
{
    defineTypeNameAndDebug(agentSolverSettings, 0);
    addToRunTimeSelectionTable(functionObject, agentSolverSettings, dictionary);
}
}


// * * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * //
void Foam::functionObjects::agentSolverSettings::writeFileHeader(Ostream& os, const word& fieldName)
{
    if (!fieldSet_.updateSelection())
    {
        return;
    }

    if (writtenHeader_)
    {
        writeBreak(file());
    }
    else
    {
        writeFile::writeHeader(os, "Solver information");
    }

    writeCommented(os, "Time");

    // write the header for the residual file
    writeTabbed(os, fieldName + "_solver");
    const word fieldBase(fieldName);

    writeTabbed(os, fieldBase + "_initial");
    writeTabbed(os, fieldBase + "_rate_median");
    writeTabbed(os, fieldBase + "_rate_max");
    writeTabbed(os, fieldBase + "_rate_min");
    writeTabbed(os, fieldBase + "_sum_iters");
    writeTabbed(os, fieldBase + "_max_iters");
    writeTabbed(os, fieldBase + "_pimple_iters");

    os << endl;

    writtenHeader_ = true;
}


void Foam::functionObjects::agentSolverSettings::writeResidualsToFile(Ostream& os, const word& fieldName)
{
    file() << token::TAB << solverName_;

    // for now only write these quantities to the file, later this will be additionally used as policy input
    file()
        << token::TAB << initialResidual_
        << token::TAB << avgAbsConRate_
        << token::TAB << maxAbsConRate_
        << token::TAB << minAbsConRate_
        << token::TAB << sumSolverIter_
        << token::TAB << maxSolverIter_
        << token::TAB << pimple_iter_;

    const word resultName(fieldName);
    setResult(resultName + "_initial", initialResidual_);
    setResult(resultName + "rate_median", avgAbsConRate_);
    setResult(resultName + "rate_max", maxAbsConRate_);
    setResult(resultName + "rate_min", minAbsConRate_);
    setResult(resultName + "_sum_iters", sumSolverIter_);
    setResult(resultName + "_max_iters", maxSolverIter_);
    setResult(resultName + "_pimple_iters", pimple_iter_);

    file() << endl;

    // reset values after writing, otherwise there are summed up throughout all following time steps
    sumSolverIter_ = 0;
    maxSolverIter_ = 0;
    maxAbsConRate_ = 0.0;
    minAbsConRate_ = 42.0;
}


void Foam::functionObjects::agentSolverSettings::computeResidualProperties(const word& fieldName)
{
    const Foam::dictionary& solverDict = fvMeshFunctionObject::mesh_.solverPerformanceDict();

    const List<SolverPerformance<scalar>> sp
    (
        solverDict.lookup(fieldName)
    );

    // same solver (GAMG) for all iterations within the time step, so just take the name of the first one
    const word& solverName = sp[0].solverName();
    const scalar& initialResidual = component(sp[0].initialResidual(), 0);

    // don't know how to set the solverName and initialResidual directly, so for now assign the values to new variables
    solverName_ = solverName;
    initialResidual_ = initialResidual;

    // we only have one GAMG call per PIMPLE iteration, otherwise computing the convergence rate would not work
    pimple_iter_ = sp.size();

    // convergence rate has one element less since we are computing the difference between two subsequent iterations
    // TODO: check if we always have only one GAMG call per pimple iter, otherwise we can't compute the convergence rate correctly
    convergenceRate_ = torch::zeros(pimple_iter_ - 1, torch::TensorOptions().dtype(torch::kFloat64));

    // loop over the residuals of each GAMG call and compute the quantities of interest
    for (label i = 0; i < pimple_iter_; i++){
        // since we are only using p, we only have one component, which is the first one
        const scalar nIterations = component(sp[i].nIterations(), 0);

        // update the max. amount of iterations of GAMG per PIMPLE call
        if (nIterations > maxSolverIter_){
            maxSolverIter_ = nIterations;
        }

        // update sum of GAMG iterations
        sumSolverIter_ += nIterations;

        // since we are computing the difference between i and i+1 element, we have one idx less
        if (i < pimple_iter_ - 1){
            // compute the convergence rate
            convergenceRate_[i] = component(sp[i].initialResidual(), 0) - component(sp[i+1].initialResidual(), 0);

            // save the max. convergence rate, magnitude because convergence rate is negative
            if (mag(convergenceRate_[i].item<double>()) > maxAbsConRate_){
                maxAbsConRate_ = mag(convergenceRate_[i].item<double>());
            }

            // save the min. convergence rate, magnitude because convergence rate is negative
            if (mag(convergenceRate_[i].item<double>()) < minAbsConRate_){
                minAbsConRate_ = mag(convergenceRate_[i].item<double>());
            }
        }
    }

    // compute the median convergence rate (correlation coefficient for median rate is higher than for avg. rate)
    avgAbsConRate_ = torch::median(convergenceRate_).abs().item<double>();
}

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::functionObjects::agentSolverSettings::agentSolverSettings
(
    const word& name,
    const Time& runTime,
    const dictionary& dict
)
:
    fvMeshFunctionObject(name, runTime, dict),

    // taken from Tomislav Maric (line 135, 136):
    // https://gitlab.com/tmaric/openfoam-ml/-/blob/master/src/aiSolutionControl/aiSolutionControl/aiSolutionControl.C?ref_type=heads#L65
    // pimpleControl(const_cast<fvMesh&>(fvMeshFunctionObject::mesh_)),
    // mesh_(fvMeshFunctionObject::mesh_),          // NOT WORKING

    writeFile(obr_, name, typeName, dict),
    fieldSet_(fvMeshFunctionObject::mesh_),
    residualFieldNames_(),
    initialised_(false),
    train_(dict.get<bool>("train")),
    policy_name_(dict.get<word>("policy")),
    policy_(torch::jit::load(policy_name_)),
    seed_(dict.get<int>("seed")),
    gen_(seed_)
{
    read(dict);
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

bool Foam::functionObjects::agentSolverSettings::read(const dictionary& dict)
{
    if (fvMeshFunctionObject::read(dict))
    {
        initialised_ = false;

        fieldSet_.read(dict);

        residualFieldNames_.clear();

        return true;
    }

    return false;
}


bool Foam::functionObjects::agentSolverSettings::execute()
{
    // we always have only the pressure field, which we want to use for our policy input, to first check if there is a
    // field available and that exactly one field is specified. Otherwise, the agent wouldn't have a policy input and crash
    // TODO: maybe better to additionally exit here
    if (fieldSet_.size() < 1)
    {
        Info << "[agentSolverSettings]: No fields given! Make sure to specify the residual field for pressure in the"
             << " controlDict. \n" << endl;
    }
    else if (fieldSet_.size() > 1)
    {
        Info << "[agentSolverSettings]: Found more than one field! Make sure to only specify one pressure field in the"
             << " controlDict. \n" << endl;
    }

    // get the field name, e.g. 'p' or 'p_rgh'; there should be only one field available
    const word& fieldName = fieldSet_.begin() -> name();

    // make sure the specified field is a scalar field, in our case some sort of pressure field. After these checks, we
    // can be sure to have the correct field and therefore don't need to check in subsequent methods
    const bool fieldExists = fvMeshFunctionObject::mesh_.foundObject<volScalarField>(fieldName);
    if (!fieldExists)
    {
        Info << "[agentSolverSettings]: specified field is either not a scalar field or doesn't exist! Make sure that"
             << "  the specified field is the correct scalar field for pressure \n" << endl;
    }

    // print some information to log file
    Info << "\n[agentSolverSettings] Predicting GAMG solver settings for the next time step.\n" << endl;

    // set the time, which has to elapse between file modification and re-reading to zero, otherwise the fvSolution file
    // would not be considered as modified, this will be active starting at the 2nd time step, compare:
    // (https://www.openfoam.com/documentation/guides/latest/api/classFoam_1_1IOobject.html#ab6ae1f9bd149d6b5097276a793d3b95f)
    IOobject::fileModificationSkew = 1e-12;

    // set the max. number of checking to 1e6, otherwise after 20 modifications (default value), the fvSolutions file
    // would always be considered as unmodified (for mixerVesselAMI, we need ~70 000 time steps to simulate 10s of physical time)
    IOobject::maxFileModificationPolls = 1e6;

    // initialize file for writing the residual data
    if (!initialised_)
    {
        writeFileHeader(file(), fieldName);

        initialised_ = true;
    }

    writeCurrentTime(file());

    // now compute the properties for the policy input of the residuals we are interested in
    computeResidualProperties(fieldName);

    // predict solver settings for next time step, for now we will only consider 'interpolateCorrection'
    predictSettings();

    // modify the solver setting dict for the pressure using the predicted settings
    modifySolverSettingsDict(fieldName);

    // write the computed properties of the residuals to the dat-file in the 'postProcessing' directory
    writeResidualsToFile(file(), fieldName);

    // force to re-read the fvSolution file (because function object is executed after checking if files have been modified)
    // compare: https://www.openfoam.com/documentation/guides/latest/api/Time_8C_source.html#l00879, line 908 / 929
    // and https://www.openfoam.com/documentation/guides/latest/api/classFoam_1_1Time.html#a6cde1928adb66791e09902caf9ce29fa
    const_cast<Time&>(fvMeshFunctionObject::mesh_.time()).readModifiedObjects();

    return true;
}


void Foam::functionObjects::agentSolverSettings::predictSettings()
{
    // this method is adopted from drlfoam, https://github.com/OFDataCommittee/drlfoam
    if (Pstream::master()) // evaluate policy only on the master
    {
        // convert all ints to doubles since the feature vector will contain doubles
        double pimple_iter = pimple_iter_;
        double maxSolverIter = maxSolverIter_;
        double sumSolverIter = sumSolverIter_;

        // create feature vector, therefore we need to convert everything to double
        std::vector<double> ft = {pimple_iter, maxSolverIter, sumSolverIter, avgAbsConRate_, initialResidual_,
                                  maxAbsConRate_, minAbsConRate_};

        const int nFeatures = ft.size();

        // fill the feature tensor for policy input
        torch::Tensor features = torch::zeros({1, nFeatures}).to(torch::kFloat64);
        for (label i = 0; i < nFeatures; i++)
        {
            if (i < 3)
            {
                features[0][i] = ft[i];
            }
            // convert the convergence rates to log, because they are a few orders smaller tha N_iter etc.
            else
            {
                features[0][i] = std::abs(std::log(ft[i]));
            }
        }

        // make prediction
        std::vector<torch::jit::IValue> policyFeatures{features};
        torch::Tensor policy_out = policy_.forward(policyFeatures).toTensor();

        if (train_)
        {
            // the 1st item is for sampling from Bernoulli-distr. for 'interpolateCorrection'
            std::bernoulli_distribution distr1(policy_out[0][0].item<double>());

            // use a discrete distribution in order to sample the smoother
            std::discrete_distribution<> distr2({policy_out[0][1].item<double>(), policy_out[0][2].item<double>(),
                                                policy_out[0][3].item<double>(), policy_out[0][4].item<double>(),
                                                policy_out[0][5].item<double>(), policy_out[0][6].item<double>()});

            action_[0] = distr1(gen_);          // action for 'interpolateCorrection'
            action_[1] = distr2(gen_);          // action for 'smoother'
        }

        else
        {
            // convert the probability for 'interpolateCorrection' to a bool
            if (policy_out[0][0].item<double>() <= 0.5)
            {
                action_[0] = 0;
            }
            else
            {
                action_[0] = 1;
            }

            // take the smoother which has the highest probability
            action_[1] = torch::argmax(torch::tensor({
                                                        policy_out[0][1].item<double>(), policy_out[0][2].item<double>(),
                                                        policy_out[0][3].item<double>(), policy_out[0][4].item<double>(),
                                                        policy_out[0][5].item<double>(), policy_out[0][6].item<double>()
                                                        })).item<int>();
        }

        // save the policy output, the execution time per time step is logged using the 'timeInfo' function object
        saveTrajectory(policy_out);
    }
}


void Foam::functionObjects::agentSolverSettings::modifySolverSettingsDict(const word& fieldName)
{
    // only modify the fvSolution file if we are on the master
    if (Pstream::master())
    {
        // map the action to the settings for 'interpolateCorrection'
        word interpolateCorrection;

        if (action_[0] == 1)
        {
           interpolateCorrection = "yes";
        }
        else
        {
            interpolateCorrection = "no";
        }

        // all available smoother for symmetric matrices (incompressible flow)
        std::vector<word> smoother = {"FDIC", "DIC", "DICGaussSeidel", "symGaussSeidel", "nonBlockingGaussSeidel",
                                      "GaussSeidel"};

        // print the new settings to log file
        Info << "\t\t\t\t\t\tNew GAMG settings: \n\t\t\t\t\t\t------------------\n\t\t\t\t\t\t\t"
             << "'interpolateCorrection' = " << interpolateCorrection << "\n\t\t\t\t\t\t\t"
             << "'smoother'              = " << smoother[action_[1]] << "\n\n" << endl;

        /* taken from Tomislav Maric (line 135, 136):
        // https://gitlab.com/tmaric/openfoam-ml/-/blob/master/src/aiSolutionControl/aiSolutionControl/aiSolutionControl.C?ref_type=heads#L65
        const fvSolution& fvSolutionDict (fvMeshFunctionObject::mesh_);
        fvSolution& fvSolutionRef = const_cast<fvSolution&>(fvSolutionDict);
        auto& solverDict = fvSolutionRef.subDict("solvers");

        IOdictionary fvSolutionDict
        (
          IOobject
           (
            "fvSolution",
            mesh_.time().system(),
            mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
           )
        );

        // read the 'fvSolutions' file and find the dict corresponding to the target quantity, e.g. 'p' or 'p_rgh'
        // we can't access the fvSolution directly, however, mesh_ is a 'volScalarField' which is derived from fvSolution
        fvSolutionDict = mesh_.lookupObject<IOdictionary>("fvSolution");

        // take the two dicts for solver settings and PIMPLE settings and save into new dict since the 'fvSolutionDict' is
        // declared as const Foam::dictionary, we can't modify it directly, so we have to save everything and write it back
        // to the file after modifications
        dictionary& solverDict = fvSolutionDict.subDict("solvers");

        // update the dict, for now only with the 'interpolateCorrection' parameter (set is altering the settings internally)
        // the values which are already present are overwritten by calling the set() method
        solverDict.subDict(fieldName).set("interpolateCorrection", interpolateCorrection);
        solverDict.subDict(fieldName).set("smoother", smoother[action_]);

        // replace the original solvers dict with the updated solver settings
        // fvSolutionDict.set("solvers", solverDict);

        // write the new solver settings to file
        // fvSolutionDict.regIOobject::write();

        // check if the smoother was successfully modified -> always 1 dt delay, because re-reading is executed prior FO
        Info << "[DEBUG] current smoother: " <<
                 fvMeshFunctionObject::mesh_.lookupObject<IOdictionary>("fvSolution").subDict("solvers").subDict(fieldName).get<word>("smoother")
                 << "\n" << endl;

        */

        // tmp work-around:
        // for now, we only modify 'interpolateCorrection' or 'smoother', so just keep everything else const.
        const word& solverSettings = "\t{\n"
                                             "\t\tsolver \tGAMG;\n"
                                             "\t\tsmoother \t" + smoother[action_[1]] + ";\n"
                                             "\t\ttolerance \t1e-06;\n"
                                             "\t\trelTol \t0.01;\n"
                                             "\t\tinterpolateCorrection \t" + interpolateCorrection + ";\n"
                                     "\t}\n";

        // update the fvSolution file
        writeFvSolutionFile(solverSettings, fieldName, 5);
        //*/
    }
}


bool Foam::functionObjects::agentSolverSettings::write()
{
    for (const word& residualName : residualFieldNames_)
    {
        const auto* residualPtr =
            fvMeshFunctionObject::mesh_.findObject<IOField<scalar>>(residualName);

        if (residualPtr)
        {
            volScalarField residual
            (
                IOobject
                (
                    residualName,
                    fvMeshFunctionObject::mesh_.time().timeName(),
                    fvMeshFunctionObject::mesh_,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE,
                    false
                ),
                fvMeshFunctionObject::mesh_,
                dimensionedScalar(dimless, Zero),
                zeroGradientFvPatchField<scalar>::typeName
            );

            residual.primitiveFieldRef() = *residualPtr;
            residual.correctBoundaryConditions();

            residual.write();
        }
    }

    return true;
}

void Foam::functionObjects::agentSolverSettings::saveTrajectory(torch::Tensor prob_out) const
{
    // taken from drlfoam, https://github.com/OFDataCommittee/drlfoam
    // for some reason values like 0.1, 0.2, ... are written out as 1, 2, ... if we write a csv file, but txt file works
    std::ifstream file("trajectory.txt");
    std::fstream trajectory("trajectory.txt", std::ios::app | std::ios::binary);
    const word t = fvMeshFunctionObject::mesh_.time().timeName();

    if(!file.good())
    {
        // write header, action0 = action for 'interpolateCorrection', action1 = action for 'smoother'
        trajectory << "t, prob0, prob1, prob2, prob3, prob4, prob5, prob6, action0, action1";
    }

    trajectory << std::setprecision(15)
               << "\n"
               << t << ", "
               // probability for 'interpolateCorrection'
               << prob_out[0][0].item<double>() << ", "

               // probabilities for 'smoother'
               << prob_out[0][1].item<double>() << ", "
               << prob_out[0][2].item<double>() << ", "
               << prob_out[0][3].item<double>() << ", "
               << prob_out[0][4].item<double>() << ", "
               << prob_out[0][5].item<double>() << ", "
               << prob_out[0][6].item<double>() << ", "
               << action_[0] << ", "
               << action_[1];
}


void Foam::functionObjects::agentSolverSettings::writeFvSolutionFile(const word& solverSettings,
                                                                     const word& fieldName, const int nParameters) const
{
    // this function modifies the fvSolution dict, however this is very inefficient and just for testing purposes
    // TODO: use IO objects / dict to modify this file more efficiently and faster, currently OF is crashing due to IO
    const fileName& systemDir = fvMeshFunctionObject::mesh_.time().system();

    // for now just use backslash until i figured out how to assemble the path independently of os
    std::ifstream file("./" + systemDir + "/" + "fvSolution");
    std::fstream fvSolutionFile("./" + systemDir + "/" + "fvSolution");

    // loop over the contents of the file, until we find our target field name
    int counter = 0;
    int start = 0;
    word line;
    word newFile;

    while (std::getline(fvSolutionFile, line))
    {
        counter++;

        // assuming the solvers dict always starts with the field of interest ,e.g. p or p_rgh
        if (line.find("solvers") != std::string::npos)
        {
            newFile += line;
            newFile += "\n";
            newFile += "{\n\t";
            newFile += fieldName;
            newFile += "\n";
            newFile += solverSettings;
            newFile += "\n";                        // account for the additional entry
            start = counter + nParameters + 4;      // len of dict + 2 lines for brackets 2 lines for line breaks
            continue;
        }

        // omit the p-dictionary of the previous time step since it is already replaced
        if (counter <= start && start > 0)
        {
            continue;
        }

        // once we omitted the original p-dict we can write the rest of the fvSolution
        else if (counter > start && start > 0)
        {
            // we need to make sure there is always only 1 new line after the dict, otherwise we would add 1 new line
            // each time step
            if (counter == start+1 && line.empty())
            {
                continue;
            }
            // Write the line to the output file
            newFile += line;
            newFile += "\n";
            continue;
        }

        // otherwise we haven't reached the solvers dict yet, so we just write the header etc. into the file
        else
        {
            newFile += line;
            newFile += "\n";
        }
    }
    fvSolutionFile.close();

    // now overwrite the fvSolution file with the new settings
    std::ofstream fvSolutionFileOut("./" + systemDir + "/" + "fvSolution");
    fvSolutionFileOut << newFile;
    fvSolutionFileOut.close();
}

// ************************************************************************* //
