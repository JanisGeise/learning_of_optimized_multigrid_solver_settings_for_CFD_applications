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
void Foam::functionObjects::agentSolverSettings::computeResidualProperties(const word& fieldName)
{
    const Foam::dictionary& solverDict = mesh_.solverPerformanceDict();

    const List<SolverPerformance<scalar>> sp
    (
        solverDict.lookup(fieldName)
    );

    // same solver (GAMG) for all iterations within the time step, so just take the name of the first one
    const word& solverName = sp[0].solverName();
    const scalar& initialResidual = component(sp[0].initialResidual(), 0);

    // we need this to access the components of the dict, however we only have one component since pressure is scalar
    const label validComponents(mesh_.validComponents<scalar>());

    // initialize the variables for storing the properties of the residuals
    int sumSolverIter = 0;
    int maxSolverIter = 0;
    scalar maxAbsConRate = 0.0;
    scalar minAbsConRate = 42.0;

    // convergence rate has one element less since we are computing the difference between two subsequent iterations
    torch::Tensor convergenceRate = torch::zeros(sp.size()-1, torch::TensorOptions().dtype(torch::kFloat64));

    // loop over the residuals of each GAMG call and compute the quantities of interest
    // TODO: check if we really only have one GAMG call per pimple iteration, otherwise we can't compute the convergence rate correctly
    for (label i = 0; i < sp.size()-1; i++){
        // since we are only using p, we only have one component, which is the first one
        const scalar nIterations = component(sp[i].nIterations(), 0);

        // update the max. amount of iterations of GAMG per PIMPLE call
        if (nIterations > maxSolverIter){
            maxSolverIter = nIterations;
        }

        // update sum of GAMG iterations
        sumSolverIter += nIterations;

        // compute the convergence rate
        convergenceRate[i] = component(sp[i].initialResidual(), 0) - component(sp[i+1].initialResidual(), 0);

        // save the max. convergence rate, magnitude because convergence rate is negative
        if (mag(convergenceRate[i].item<double>()) > maxAbsConRate){
            maxAbsConRate = mag(convergenceRate[i].item<double>());
        }

        // save the min. convergence rate, magnitude because convergence rate is negative
        if (mag(convergenceRate[i].item<double>()) < minAbsConRate){
            minAbsConRate = mag(convergenceRate[i].item<double>());
        }
    }

    // for simplicity, for now just compute the avg. convergence rate instead of median
    scalar avgAbsConRate = torch::mean(convergenceRate).abs().item<double>();

    file() << token::TAB << solverName;

    // for now only write these quantities to the file, later this will be additionally used as policy input
    if (component(validComponents, 0) != -1)
    {
        // we only have one GAMG call per PIMPLE iteration, otherwise computing the convergence rate would not work
        const int pimple_iter = sp.size();

        file()
            << token::TAB << initialResidual
            << token::TAB << avgAbsConRate
            << token::TAB << maxAbsConRate
            << token::TAB << minAbsConRate
            << token::TAB << sumSolverIter
            << token::TAB << maxSolverIter
            << token::TAB << pimple_iter;

        const word resultName(fieldName);
        setResult(resultName + "_initial", initialResidual);
        setResult(resultName + "rate_avg", avgAbsConRate);
        setResult(resultName + "rate_max", maxAbsConRate);
        setResult(resultName + "rate_min", minAbsConRate);
        setResult(resultName + "_sum_iters", sumSolverIter);
        setResult(resultName + "_max_iters", maxSolverIter);
        setResult(resultName + "_pimple_iters", pimple_iter);
    }
}


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
        writeHeader(os, "Solver information");
    }

    writeCommented(os, "Time");

    // write the header for the residual file
    writeTabbed(os, fieldName + "_solver");
    const word fieldBase(fieldName);

    writeTabbed(os, fieldBase + "_initial");
    writeTabbed(os, fieldBase + "_rate_abs");
    writeTabbed(os, fieldBase + "_rate_max");
    writeTabbed(os, fieldBase + "_rate_min");
    writeTabbed(os, fieldBase + "_sum_iters");
    writeTabbed(os, fieldBase + "_max_iters");
    writeTabbed(os, fieldBase + "_pimple_iters");

    os << endl;

    writtenHeader_ = true;
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
    writeFile(obr_, name, typeName, dict),
    fieldSet_(mesh_),
    residualFieldNames_(),
    initialised_(false)
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
    // field available and that exactly one field is specified. Otherwise, the agent wouldn't have a policy input crash
    // TODO: maybe better to additionally exit here
    if (fieldSet_.size() < 1)
    {
        Info << "[agentSolverSettings]: No fields given! Make sure to specify the residual field for pressure in the"
             << " controlDict." << endl;
    }
    else if (fieldSet_.size() > 1)
    {
        Info << "[agentSolverSettings]: Found more than one field! Make sure to only specify one pressure field in the"
             << " controlDict." << endl;
    }

    // get the field name, e.g. 'p' or 'p_rgh'; there should be only one field available
    const word& fieldName = fieldSet_.begin() -> name();

    // make sure the specified field is a scalar field, in our case some sort of pressure field. After these checks, we
    // can be sure to have the correct field and therefore don't need to check in subsequent methods
    const bool fieldExists = mesh_.foundObject<volScalarField>(fieldName);
    if (!fieldExists)
    {
        Info << "[agentSolverSettings]: specified field is either not a scalar field or doesn't exist! Make sure that"
             << "  the specified field is the correct scalar field for pressure" << endl;
    }

    // Note: delaying initialisation until after first iteration so that
    // we can find wildcard fields
    if (!initialised_)
    {
        writeFileHeader(file(), fieldName);

        initialised_ = true;
    }

    writeCurrentTime(file());

    // not compute the properties for the policy input of the residuals we are interested in
    computeResidualProperties(fieldName);

    file() << endl;

    return true;
}


bool Foam::functionObjects::agentSolverSettings::write()
{
    for (const word& residualName : residualFieldNames_)
    {
        const auto* residualPtr =
            mesh_.findObject<IOField<scalar>>(residualName);

        if (residualPtr)
        {
            volScalarField residual
            (
                IOobject
                (
                    residualName,
                    mesh_.time().timeName(),
                    mesh_,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE,
                    false
                ),
                mesh_,
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


// ************************************************************************* //
